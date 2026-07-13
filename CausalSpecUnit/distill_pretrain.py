"""
DistilHuBERT-style knowledge-distillation pretraining.

Distills a frozen, released HuBERT Base / wav2vec 2.0 Base teacher (95M) into
the compact 9M SqueezeFormer-XS encoder on the unlabeled 960h corpus. The
student predicts the teacher's multi-layer features from its own final encoder
output, using the DistilHuBERT L1 + cosine objective (no input masking).

This is a drop-in replacement for the SSL pretraining stage: the resulting
checkpoint stores encoder weights under ``encoder.*``, so downstream CTC
fine-tuning is the *unchanged* train_ctc.py:

    torchrun ... CausalSpecUnit/train_ctc.py \
        --ssl-checkpoint outputs/causal_specunit/distill/checkpoint_stepNNNNNN \
        --variant xs --train-subset-hours 10 ...

The projection heads (student->teacher-dim) are training-only; train_ctc.py's
load_ssl_encoder reads only ``encoder.*`` and ignores them.

Rationale
---------
The reviewer asked for a competitive same-footprint baseline: knowledge
distillation from a larger pretrained teacher into a compact student. This is
the canonical DistilHuBERT recipe (multi-layer feature prediction, L1+cosine),
so the comparison cannot be dismissed as a weakened baseline. GPU-hours for
this stage are directly comparable to our SSL pretraining budget.
"""

import argparse
import contextlib
import json
import math
import os
import shutil
import socket
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from CausalSpecUnit.common import (
    TRAIN_SPLITS,
    barrier,
    build_extended_noam_scheduler,
    cleanup_distributed,
    is_main_process,
    load_checkpoint,
    print0,
    save_checkpoint,
    setup_distributed,
    unwrap_model,
)
from CausalSpecUnit.data import DistillDataset, LengthBucketSampler, collate_distill
from CausalSpecUnit.model import SQUEEZEFORMER_VARIANTS, build_encoder, get_encoder_dim
from CausalSpecUnit.teacher import SUPPORTED_TEACHERS, build_teacher


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


class DistillStudent(nn.Module):
    """SqueezeFormer-XS encoder + one linear prediction head per teacher layer.

    The encoder is exactly the one used for SSL/CTC, so its weights are saved
    under ``encoder.*`` and load cleanly into CausalSpecUnitCTC downstream.
    Each ``proj_heads[i]`` maps the shared final encoder output to the teacher
    feature dim, predicting teacher layer ``i``. Heads are discarded after
    distillation.
    """

    def __init__(self, variant, teacher_dim, num_heads, layer_drop_p=0.0):
        super().__init__()
        self.variant = variant
        self.encoder = build_encoder(variant, layer_drop_p=layer_drop_p)
        self.encoder_dim = get_encoder_dim(variant)
        self.proj_heads = nn.ModuleList(
            [nn.Linear(self.encoder_dim, teacher_dim) for _ in range(num_heads)]
        )

    def forward(self, mel, lengths):
        encoded, out_lengths, _ = self.encoder(mel, lengths)
        preds = [head(encoded) for head in self.proj_heads]
        return preds, out_lengths


def distill_loss(preds, student_lengths, teacher_feats, teacher_lengths, cosine_weight):
    """DistilHuBERT feature loss: mean over layers of [L1 + w*(1 - cosine)].

    Computed only over valid (non-padded) frames, aligned to the shorter of the
    student and teacher time axes per layer.
    """
    device = preds[0].device
    # Base term stays connected to every head's graph but contributes exactly 0,
    # so backward() is safe even if all frames are masked out in a batch.
    total = sum(p.sum() * 0.0 for p in preds)
    l1_acc = 0.0
    cos_acc = 0.0
    n = len(preds)
    for pred, tgt in zip(preds, teacher_feats):
        T = min(pred.size(1), tgt.size(1))
        pred = pred[:, :T]
        tgt = tgt[:, :T]
        valid_len = torch.minimum(student_lengths, teacher_lengths).clamp(max=T)
        frame_mask = torch.arange(T, device=device)[None, :] < valid_len[:, None]  # [B, T]
        if not frame_mask.any():
            continue
        sel = frame_mask.unsqueeze(-1).expand_as(pred)
        pv = pred[sel].view(-1, pred.size(-1))
        tv = tgt[sel].view(-1, tgt.size(-1))
        l1 = F.l1_loss(pv, tv)
        cos = 1.0 - F.cosine_similarity(pv, tv, dim=-1).mean()
        total = total + l1 + cosine_weight * cos
        l1_acc += float(l1.detach().item())
        cos_acc += float(cos.detach().item())
    return total / max(n, 1), l1_acc / max(n, 1), cos_acc / max(n, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="dataset/datasets/librispeech/LibriSpeech")
    p.add_argument("--cmvn-path", type=str, default="outputs/causal_specunit/targets/cmvn.pt")
    p.add_argument("--output-dir", type=str, default="outputs/causal_specunit/distill")
    p.add_argument("--mel-cache-dir", type=str, default=None,
                   help="Optional directory of precomputed CMVN log-mel tensors (reuses the SSL mel cache).")
    p.add_argument("--splits", nargs="+", default=None,
                   help="Override training splits (default: TRAIN_SPLITS = full 960h).")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint directory to resume from.")
    # ---- memory / throughput ----
    p.add_argument("--max-duration-sec", type=float, default=None,
                   help="Drop utterances longer than this many seconds. Bounds the frozen "
                        "teacher's activation memory (its forward over long waveforms is the "
                        "main OOM risk). None keeps all utterances.")
    p.add_argument("--bucket-sampler", action="store_true",
                   help="Batch utterances of similar length together to cut padding waste and "
                        "peak memory. Requires a one-time duration scan (cached via --durations-cache).")
    p.add_argument("--durations-cache", type=str, default=None,
                   help="JSON uid->seconds cache for --max-duration-sec / --bucket-sampler, so the "
                        "duration scan runs only once across runs.")
    # ---- teacher ----
    p.add_argument("--teacher", type=str, default="hubert_base", choices=list(SUPPORTED_TEACHERS))
    p.add_argument("--teacher-layers", type=int, nargs="+", default=[3, 7, 11],
                   help="0-indexed teacher transformer layers to distill. Default 3 7 11 "
                        "(= HuBERT layers 4/8/12, DistilHuBERT's choice).")
    p.add_argument("--no-teacher-downsample", dest="teacher_downsample", action="store_false",
                   help="Disable 2x teacher time downsampling. Keep enabled unless the student "
                        "output rate matches the teacher's ~50 fps.")
    p.set_defaults(teacher_downsample=True)
    p.add_argument("--cosine-weight", type=float, default=1.0,
                   help="Weight on the (1 - cosine) term relative to L1 (DistilHuBERT uses ~1.0).")
    # ---- student / optim ----
    p.add_argument("--variant", type=str, default="xs", choices=list(SQUEEZEFORMER_VARIANTS))
    p.add_argument("--layer-drop-p", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dataloader-timeout", type=int, default=120)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--peak-epochs", type=int, default=20)
    p.add_argument("--noam-decay-rate", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps.")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--keep-checkpoints", type=int, default=5)
    p.add_argument("--progress", choices=["on", "off"], default="on")
    return p.parse_args()


def cleanup_checkpoints(output_dir, keep):
    if keep <= 0 or not os.path.isdir(output_dir):
        return
    ckpts = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint_step"):
            try:
                step = int(name.replace("checkpoint_step", ""))
            except ValueError:
                continue
            ckpts.append((step, name))
    ckpts.sort()
    for _, name in ckpts[:-keep]:
        shutil.rmtree(os.path.join(output_dir, name), ignore_errors=True)


def count_parameters(model):
    model = unwrap_model(model)
    total = sum(p.numel() for p in model.parameters())
    encoder = sum(p.numel() for p in model.encoder.parameters())
    return {"total": total, "encoder": encoder}


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    os.makedirs(args.output_dir, exist_ok=True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    dataset = DistillDataset(
        data_root=args.data_root,
        splits=args.splits if args.splits else TRAIN_SPLITS,
        cmvn_path=args.cmvn_path,
        mel_cache_dir=args.mel_cache_dir,
        max_duration_sec=args.max_duration_sec,
        compute_durations=args.bucket_sampler,
        durations_cache=args.durations_cache,
    )
    if args.bucket_sampler:
        if dataset.durations is None:
            raise RuntimeError("--bucket-sampler requested but durations were not computed.")
        sampler = LengthBucketSampler(
            lengths=dataset.durations,
            batch_size=args.batch_size,
            num_replicas=max(1, world_size),
            rank=rank,
        )
    elif world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None
    worker_kwargs = {}
    if args.workers > 0:
        worker_kwargs = {"persistent_workers": True, "prefetch_factor": max(1, args.prefetch_factor)}
    dataloader_timeout = args.dataloader_timeout if args.workers > 0 else 0
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_distill,
        pin_memory=True,
        drop_last=True,
        timeout=dataloader_timeout,
        **worker_kwargs,
    )

    # Frozen teacher — never wrapped in DDP (no gradients, no param sync).
    teacher = build_teacher(
        name=args.teacher,
        layers=tuple(args.teacher_layers),
        downsample=args.teacher_downsample,
    ).to(device)
    teacher.eval()

    student = DistillStudent(
        variant=args.variant,
        teacher_dim=teacher.output_dim,
        num_heads=len(args.teacher_layers),
        layer_drop_p=args.layer_drop_p,
    ).to(device)

    if world_size > 1:
        ddp_find_unused = bool(args.layer_drop_p > 0.0)
        student = DDP(
            student,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused,
        )

    metrics_path = os.path.join(args.output_dir, "distill_metrics.jsonl")
    run_info_path = os.path.join(args.output_dir, "distill_run_info.json")
    if is_main_process(rank):
        run_info = {
            "argv": sys.argv,
            "args": vars(args),
            "parameter_counts": count_parameters(student),
            "teacher": args.teacher,
            "teacher_layers": args.teacher_layers,
            "teacher_dim": teacher.output_dim,
            "world_size": world_size,
            "device": str(device),
            "hostname": socket.gethostname(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(run_info_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2, sort_keys=True)
        append_jsonl(metrics_path, {"event": "run_start", **run_info})

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(loader) / max(1, args.grad_accum_steps))
    scheduler = build_extended_noam_scheduler(
        optimizer, steps_per_epoch,
        warmup_epochs=args.warmup_epochs, peak_epochs=args.peak_epochs, decay_rate=args.noam_decay_rate,
    )
    optimizer_steps = 0
    start_epoch = 1
    if args.resume:
        ckpt = load_checkpoint(args.resume, unwrap_model(student), optimizer, scheduler, device=device)
        optimizer_steps = int(ckpt.get("optimizer_steps", 0))
        start_epoch = int(ckpt.get("epoch", 1)) + 1
        print0(rank, f"[distill] resumed from {args.resume} | opt_step={optimizer_steps} start_epoch={start_epoch}")

    print0(
        rank,
        f"Distill {args.teacher} (layers={args.teacher_layers}) -> SqueezeFormer-{args.variant.upper()} | "
        f"train={len(dataset)} utt | world={world_size} | "
        f"effective_batch={args.batch_size * world_size * args.grad_accum_steps} | "
        f"lr={args.lr:g} warmup={args.warmup_epochs} hold={args.peak_epochs}",
    )
    job_start = time.time()

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if args.max_steps is not None and optimizer_steps >= args.max_steps:
                break
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            student.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = total_l1 = total_cos = 0.0
            n_batches = 0
            show = args.progress == "on" and is_main_process(rank)
            bar = tqdm(loader, desc=f"distill {epoch:03d}", leave=False, disable=not show)

            for step, batch in enumerate(bar, start=1):
                if args.max_train_batches is not None and step > args.max_train_batches:
                    break
                mel, lengths, waveforms, wav_lengths = batch
                mel = mel.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                waveforms = waveforms.to(device, non_blocking=True)
                wav_lengths = wav_lengths.to(device, non_blocking=True)

                sync_step = step % max(1, args.grad_accum_steps) == 0 or step == len(loader)
                window_start = ((step - 1) // max(1, args.grad_accum_steps)) * max(1, args.grad_accum_steps) + 1
                window_end = min(window_start + max(1, args.grad_accum_steps) - 1, len(loader))
                actual_accum_steps = window_end - window_start + 1
                sync_context = student.no_sync if isinstance(student, DDP) and not sync_step else contextlib.nullcontext

                with sync_context():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        with torch.no_grad():
                            teacher_feats, teacher_lengths = teacher(waveforms, wav_lengths)
                        preds, out_lengths = student(mel, lengths)
                        # Feature loss in fp32 for numerical stability of cosine.
                        teacher_feats = [t.float() for t in teacher_feats]
                        preds = [p.float() for p in preds]
                        loss_full, l1_val, cos_val = distill_loss(
                            preds, out_lengths, teacher_feats, teacher_lengths, args.cosine_weight,
                        )
                        loss = loss_full / actual_accum_steps
                    loss.backward()

                grad_norm_value = None
                if sync_step:
                    grad_norm = nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                    grad_norm_value = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1

                loss_val = loss_full.detach().float().item()
                total_loss += loss_val
                total_l1 += l1_val
                total_cos += cos_val
                n_batches += 1
                if show:
                    bar.set_postfix(
                        loss=f"{loss_val:.3f}", l1=f"{l1_val:.3f}", cos=f"{cos_val:.3f}",
                        lr=f"{scheduler.get_last_lr()[0]:.1e}",
                        step=f"{optimizer_steps}/{args.max_steps or '?'}", refresh=False,
                    )
                if args.log_every > 0 and is_main_process(rank) and step % args.log_every == 0:
                    append_jsonl(metrics_path, {
                        "event": "train_step",
                        "epoch": epoch,
                        "batch": step,
                        "optimizer_step": optimizer_steps,
                        "loss": loss_val,
                        "l1": l1_val,
                        "cosine_dist": cos_val,
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm_value,
                        "elapsed_hours": (time.time() - job_start) / 3600,
                    })
                if args.max_steps is not None and optimizer_steps >= args.max_steps:
                    break

            avg = total_loss / max(n_batches, 1)
            avg_l1 = total_l1 / max(n_batches, 1)
            avg_cos = total_cos / max(n_batches, 1)
            elapsed = time.time() - job_start
            print0(
                rank,
                f"[distill] epoch={epoch:03d} opt_step={optimizer_steps} loss={avg:.4f} "
                f"l1={avg_l1:.4f} cos_dist={avg_cos:.4f} elapsed={elapsed/3600:.2f}h",
            )
            if is_main_process(rank):
                append_jsonl(metrics_path, {
                    "event": "epoch_end",
                    "epoch": epoch,
                    "optimizer_step": optimizer_steps,
                    "loss": avg,
                    "l1": avg_l1,
                    "cosine_dist": avg_cos,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_hours": elapsed / 3600,
                })
            barrier()
            if is_main_process(rank) and (
                epoch % args.save_every == 0
                or (args.max_steps is not None and optimizer_steps >= args.max_steps)
                or epoch == args.epochs
            ):
                save_checkpoint(
                    os.path.join(args.output_dir, f"checkpoint_step{optimizer_steps:06d}"),
                    student,
                    optimizer,
                    scheduler,
                    epoch,
                    extra={"distill_loss": avg, "optimizer_steps": optimizer_steps},
                )
                cleanup_checkpoints(args.output_dir, args.keep_checkpoints)
            barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
