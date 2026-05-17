import argparse
import json
import math
import os
import shutil
import socket
import sys
import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from SqueezeFormer.train import build_tokenizer
from CausalSpecUnit.common import (
    DEV_SPLIT,
    TRAIN_SPLITS,
    barrier,
    build_extended_noam_scheduler,
    cleanup_distributed,
    is_main_process,
    print0,
    save_checkpoint,
    setup_distributed,
)
from CausalSpecUnit.data import CTCSpecDataset, collate_ctc, collate_eval
from CausalSpecUnit.model import CausalSpecUnitCTC


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def count_parameters(model):
    target = model.module if hasattr(model, "module") else model
    total = sum(p.numel() for p in target.parameters())
    trainable = sum(p.numel() for p in target.parameters() if p.requires_grad)
    encoder = sum(p.numel() for p in target.encoder.parameters())
    return {"total": total, "trainable": trainable, "encoder": encoder}


def current_lrs(optimizer):
    lrs = {}
    for idx, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group_{idx}")
        lrs[name] = group["lr"]
    return lrs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="dataset/datasets/librispeech/LibriSpeech")
    p.add_argument("--cmvn-path", type=str, default="outputs/causal_specunit/targets/cmvn.pt")
    p.add_argument("--ssl-checkpoint", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="outputs/causal_specunit/ctc")
    p.add_argument("--tokenizer-path", type=str, default="dataset/bpe128.model")
    p.add_argument("--train-splits", nargs="+", default=TRAIN_SPLITS,
                   help="Training splits to use. Defaults to full LibriSpeech train-960.")
    p.add_argument("--train-subset-hours", type=float, default=None,
                   help="Use a reproducible random subset with approximately this many audio hours.")
    p.add_argument("--train-subset-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dataloader-timeout", type=int, default=120)
    p.add_argument("--variant", type=str, default="xs", choices=["xs", "s", "sm", "m", "ml", "l"])
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--encoder-lr", type=float, default=None,
                   help="Optional peak LR for encoder parameters, useful for SSL fine-tuning.")
    p.add_argument("--head-lr", type=float, default=None,
                   help="Optional peak LR for non-encoder parameters, including the CTC head.")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--peak-epochs", type=int, default=160,
                   help="Number of epochs to hold the peak LR after warmup before Noam decay.")
    p.add_argument("--noam-decay-rate", type=float, default=1.0)
    p.add_argument("--eval-split", type=str, default=DEV_SPLIT)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--keep-checkpoints", type=int, default=5)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--progress", choices=["on", "off"], default="on")
    return p.parse_args()


def cleanup_epoch_checkpoints(output_dir, keep):
    if keep <= 0 or not os.path.isdir(output_dir):
        return
    ckpts = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint_ep"):
            try:
                epoch = int(name.split("_ep")[-1])
            except ValueError:
                continue
            ckpts.append((epoch, name))
    ckpts.sort()
    for _, name in ckpts[:-keep]:
        shutil.rmtree(os.path.join(output_dir, name), ignore_errors=True)


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, start=1):
            old = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = old
    return dp[-1]


def greedy_decode(log_probs, lengths, blank_id):
    preds = log_probs.argmax(dim=-1)
    decoded = []
    for b in range(preds.size(0)):
        seq, prev = [], None
        for tok in preds[b, :lengths[b]].tolist():
            if tok != prev:
                seq.append(tok)
            prev = tok
        decoded.append([t for t in seq if t != blank_id])
    return decoded


@torch.no_grad()
def evaluate(model, loader, tokenizer, blank_id, ctc_loss, device, rank, show_progress=True):
    model.eval()
    total_word_errors = 0
    total_ref_words = 0
    total_loss = 0.0
    total_batches = 0
    example = None
    bar = tqdm(loader, desc="eval", leave=False, disable=not (show_progress and is_main_process(rank)))
    for mel, lengths, labels, label_lengths, transcripts in bar:
        mel = mel.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            log_probs, output_lengths = model(mel, lengths)
            loss = ctc_loss(log_probs.transpose(0, 1), labels, output_lengths, label_lengths)
        total_loss += loss.detach().float().item()
        total_batches += 1
        hyps = [tokenizer.decode(x).lower().strip() for x in greedy_decode(log_probs, output_lengths, blank_id)]
        refs = [x.lower().strip() for x in transcripts]
        for hyp, ref in zip(hyps, refs):
            hw, rw = hyp.split(), ref.split()
            total_word_errors += edit_distance(hw, rw)
            total_ref_words += max(len(rw), 1)
            if example is None:
                example = (hyp, ref)
    return {
        "loss": total_loss / max(total_batches, 1),
        "wer": total_word_errors / max(total_ref_words, 1),
        "example": example,
    }


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    os.makedirs(args.output_dir, exist_ok=True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    tokenizer = build_tokenizer(args.tokenizer_path)
    blank_id = tokenizer.pad_token_id
    train_dataset = CTCSpecDataset(
        args.data_root,
        args.train_splits,
        tokenizer,
        cmvn_path=args.cmvn_path,
        train_split=True,
        max_hours=args.train_subset_hours,
        subset_seed=args.train_subset_seed,
    )
    dev_dataset = CTCSpecDataset(args.data_root, [args.eval_split], tokenizer, cmvn_path=args.cmvn_path, train_split=False)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    dev_sampler = DistributedSampler(dev_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    worker_kwargs = {"persistent_workers": True, "prefetch_factor": 4} if args.workers > 0 else {}
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=collate_ctc,
        pin_memory=True,
        drop_last=True,
        timeout=args.dataloader_timeout,
        **worker_kwargs,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        sampler=dev_sampler,
        num_workers=args.workers,
        collate_fn=collate_eval,
        pin_memory=True,
        timeout=args.dataloader_timeout,
        **worker_kwargs,
    )

    model = CausalSpecUnitCTC(
        vocab_size=tokenizer.vocab_size,
        variant=args.variant,
    )
    if args.ssl_checkpoint:
        missing, unexpected = model.load_ssl_encoder(args.ssl_checkpoint, map_location="cpu")
        print0(rank, f"Loaded SSL encoder from {args.ssl_checkpoint} | missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    metrics_path = os.path.join(args.output_dir, "ctc_metrics.jsonl")
    run_info_path = os.path.join(args.output_dir, "ctc_run_info.json")
    if is_main_process(rank):
        run_info = {
            "event": "run_start",
            "argv": sys.argv,
            "args": vars(args),
            "world_size": world_size,
            "device": str(device),
            "hostname": socket.gethostname(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parameter_counts": count_parameters(model),
            "train_utterances": len(train_dataset),
            "dev_utterances": len(dev_dataset),
            "effective_batch": args.batch_size * world_size * args.grad_accum_steps,
            "ssl_initialized": bool(args.ssl_checkpoint),
            "train_splits": args.train_splits,
            "train_audio_hours": getattr(train_dataset, "audio_hours", None),
        }
        with open(run_info_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2, sort_keys=True)
        append_jsonl(metrics_path, run_info)

    opt_model = model.module if hasattr(model, "module") else model
    if args.encoder_lr is not None or args.head_lr is not None:
        encoder_lr = args.encoder_lr if args.encoder_lr is not None else args.lr
        head_lr = args.head_lr if args.head_lr is not None else args.lr
        encoder_param_ids = {id(p) for p in opt_model.encoder.parameters()}
        head_params = [p for p in opt_model.parameters() if id(p) not in encoder_param_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": opt_model.encoder.parameters(), "lr": encoder_lr, "name": "encoder"},
                {"params": head_params, "lr": head_lr, "name": "head"},
            ],
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=args.weight_decay,
        )
        print0(rank, f"LR groups: encoder={encoder_lr:g} | head={head_lr:g}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=args.weight_decay,
        )
    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
    scheduler = build_extended_noam_scheduler(
        optimizer,
        steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        peak_epochs=args.peak_epochs,
        decay_rate=args.noam_decay_rate,
    )
    ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    best_wer = float("inf")
    optimizer_steps = 0
    run_start = time.time()
    hours_note = ""
    if getattr(train_dataset, "audio_hours", None) is not None:
        hours_note = f" train_hours={train_dataset.audio_hours:.2f}"
    print0(
        rank,
        f"CausalSpecUnit CTC | train={len(train_dataset)} dev={len(dev_dataset)} "
        f"world={world_size} effective_batch={args.batch_size * world_size * args.grad_accum_steps}"
        f"{hours_note} | warmup={args.warmup_epochs} hold={args.peak_epochs} decay={args.noam_decay_rate:g}",
    )

    try:
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            n_batches = 0
            show = args.progress == "on" and is_main_process(rank)
            bar = tqdm(train_loader, desc=f"CTC {epoch:03d}", leave=False, disable=not show)
            for step, batch in enumerate(bar, start=1):
                if args.max_train_batches is not None and step > args.max_train_batches:
                    break
                mel, lengths, labels, label_lengths = batch
                mel = mel.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                label_lengths = label_lengths.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    log_probs, output_lengths = model(mel, lengths)
                    loss = ctc_loss(log_probs.transpose(0, 1), labels, output_lengths, label_lengths)
                    loss = loss / max(1, args.grad_accum_steps)
                loss.backward()
                sync_step = step % max(1, args.grad_accum_steps) == 0 or step == len(train_loader)
                grad_norm_value = None
                if sync_step:
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    grad_norm_value = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                loss_val = loss.detach().float().item() * max(1, args.grad_accum_steps)
                total_loss += loss_val
                n_batches += 1
                if show:
                    bar.set_postfix(loss=f"{loss_val:.3f}", avg=f"{total_loss/max(n_batches,1):.3f}", lr=f"{scheduler.get_last_lr()[0]:.1e}", refresh=False)
                if args.log_every > 0 and is_main_process(rank) and step % args.log_every == 0:
                    append_jsonl(metrics_path, {
                        "event": "train_step",
                        "epoch": epoch,
                        "batch": step,
                        "batches_per_epoch": len(train_loader),
                        "optimizer_step": optimizer_steps,
                        "train_loss": loss_val,
                        "train_loss_avg": total_loss / max(n_batches, 1),
                        "lr": scheduler.get_last_lr()[0],
                        "lrs": current_lrs(optimizer),
                        "grad_norm": grad_norm_value,
                        "elapsed_hours": (time.time() - run_start) / 3600,
                    })

            avg_loss = total_loss / max(n_batches, 1)
            should_eval = args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs)
            if should_eval and dev_sampler is not None:
                dev_sampler.set_epoch(epoch)
            if should_eval:
                metrics = evaluate(model, dev_loader, tokenizer, blank_id, ctc_loss, device, rank, show_progress=show)
                if is_main_process(rank):
                    best_wer = min(best_wer, metrics["wer"])
                    hyp, ref = metrics["example"] or ("", "")
                    tqdm.write(
                        f"[ctc] epoch={epoch:03d} train_loss={avg_loss:.4f} dev_loss={metrics['loss']:.4f} "
                        f"wer={metrics['wer']:.2%} best={best_wer:.2%}\nREF: {ref}\nHYP: {hyp}"
                    )
                    append_jsonl(metrics_path, {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "optimizer_step": optimizer_steps,
                        "train_loss": avg_loss,
                        "dev_loss": metrics["loss"],
                        "wer": metrics["wer"],
                        "best_wer": best_wer,
                        "lr": scheduler.get_last_lr()[0],
                        "lrs": current_lrs(optimizer),
                        "elapsed_hours": (time.time() - run_start) / 3600,
                        "example_ref": ref,
                        "example_hyp": hyp,
                    })
                    if metrics["wer"] <= best_wer:
                        save_checkpoint(
                            os.path.join(args.output_dir, "checkpoint_best"),
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            extra={"best_wer": best_wer, "optimizer_steps": optimizer_steps},
                        )
            else:
                print0(rank, f"[ctc] epoch={epoch:03d} train_loss={avg_loss:.4f}")
                if is_main_process(rank):
                    append_jsonl(metrics_path, {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "optimizer_step": optimizer_steps,
                        "train_loss": avg_loss,
                        "dev_loss": None,
                        "wer": None,
                        "best_wer": best_wer if math.isfinite(best_wer) else None,
                        "lr": scheduler.get_last_lr()[0],
                        "lrs": current_lrs(optimizer),
                        "elapsed_hours": (time.time() - run_start) / 3600,
                    })
            barrier()
            if epoch % args.save_every == 0 and is_main_process(rank):
                save_checkpoint(
                    os.path.join(args.output_dir, f"checkpoint_ep{epoch:03d}"),
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    extra={"best_wer": best_wer, "optimizer_steps": optimizer_steps},
                )
                cleanup_epoch_checkpoints(args.output_dir, args.keep_checkpoints)
            barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
