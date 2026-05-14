import argparse
import socket
import sys
import time
import contextlib
import json
import math
import os
import shutil

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record
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
from CausalSpecUnit.data import SpecUnitDataset, collate_ssl
from CausalSpecUnit.model import CausalSpecUnitSSL


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="dataset/datasets/librispeech/LibriSpeech")
    p.add_argument("--targets-dir", type=str, default="outputs/causal_specunit/targets")
    p.add_argument("--output-dir", type=str, default="outputs/causal_specunit/pretrain")
    p.add_argument("--mel-cache-dir", type=str, default=None,
                   help="Optional directory of precomputed CMVN log-mel tensors.")
    p.add_argument("--splits", nargs="+", default=None,
                   help="Override training splits (default: TRAIN_SPLITS constant).")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint directory to resume from.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dataloader-timeout", type=int, default=120)
    p.add_argument("--prefetch-factor", type=int, default=2,
                   help="Number of batches prefetched per DataLoader worker.")
    p.add_argument("--variant", type=str, default="xs", choices=["xs", "s", "sm", "m", "ml", "l"])
    p.add_argument("--chunk-size", type=int, default=4,
                   help="Frames per target chunk. Must match target generation.")
    p.add_argument("--chunk-stride", type=int, default=4,
                   help="Frame stride between target chunks. Must match target generation.")
    p.add_argument("--mask-prob", type=float, default=0.065,
                   help="HuBERT-style probability of starting a time mask span over target steps.")
    p.add_argument("--mask-length", type=int, default=10,
                   help="HuBERT-style mask span length in target steps.")
    p.add_argument("--mask-value", type=float, default=0.0,
                   help="Value used to replace masked CMVN-normalized spectrogram frames.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--peak-epochs", type=int, default=20)
    p.add_argument("--noam-decay-rate", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-safe-grad-norm", type=float, default=200.0,
                   help="Skip optimizer/scheduler step if the pre-clipping grad norm exceeds this value. Use 0 to disable.")
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--keep-checkpoints", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Stop after this many optimizer steps.")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--progress", choices=["on", "off"], default="on")
    p.add_argument("--bucket-sampler", action="store_true",
                   help="Sort batches by sequence length to reduce padding waste.")
    p.add_argument("--compile", action="store_true",
                   help="Apply torch.compile to the model for faster training.")
    p.add_argument("--trace-startup", action="store_true",
                   help="Print rank-aware startup traces to locate hangs before/at the first batch.")
    p.add_argument("--trace-every", type=int, default=0,
                   help="Print rank-aware batch traces every N steps. Use 0 to disable.")
    return p.parse_args()


def trace(enabled, rank, message):
    if not enabled:
        return
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    host = socket.gethostname()
    print(f"[trace {now}] host={host} rank={rank} {message}", file=sys.stderr, flush=True)


def validate_target_metadata(targets_dir, args):
    metadata_path = os.path.join(targets_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Missing target metadata: {metadata_path}. Regenerate targets with the current generate_targets.py."
        )
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    expected = {
        "chunk_size": args.chunk_size,
        "chunk_stride": args.chunk_stride,
        "k_coarse": 100,
        "k_fine": 500,
    }
    for key, value in expected.items():
        if int(metadata[key]) != int(value):
            raise ValueError(
                f"Target metadata mismatch for {key}: targets have {metadata[key]}, "
                f"but pretraining was launched with {value}."
            )
    return metadata


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


_warned_empty_mask = False


def masked_unit_ce(logits, targets, masked_positions):
    """CE over only masked positions. Pads use target=-100 and are ignored."""
    global _warned_empty_mask
    t = min(logits.size(1), targets.size(1), masked_positions.size(1))
    logits = logits[:, :t]
    targets = targets[:, :t]
    masked_positions = masked_positions[:, :t] & targets.ne(-100)
    if not masked_positions.any():
        if not _warned_empty_mask:
            print("[ssl warn] masked_unit_ce: no masked positions in batch — loss is zero; check mask_prob/mask_length", flush=True)
            _warned_empty_mask = True
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[masked_positions], targets[masked_positions])


_warned_align_crop = False


def align_ssl_tensors(coarse_logits, fine_logits, z100, z500, masked_positions):
    """Crop model outputs, clean labels, and mask to one shared target length."""
    global _warned_align_crop
    lengths = {
        "coarse_logits": coarse_logits.size(1),
        "fine_logits": fine_logits.size(1),
        "z100": z100.size(1),
        "z500": z500.size(1),
        "mask": masked_positions.size(1),
    }
    t = min(lengths.values())
    tmax = max(lengths.values())
    if tmax - t > 4 and not _warned_align_crop:
        print(f"[ssl warn] align_ssl_tensors: cropping {tmax - t} frames (lengths={lengths}); "
              "check chunk_size/chunk_stride vs model downsampling", flush=True)
        _warned_align_crop = True
    return (
        coarse_logits[:, :t],
        fine_logits[:, :t],
        z100[:, :t],
        z500[:, :t],
        masked_positions[:, :t],
    )


def make_hubert_mask(target_lengths, max_targets, mask_prob, mask_length, device):
    """HuBERT-style target mask, fully vectorized (no Python loop over batch items)."""
    B = target_lengths.size(0)
    mask_length = max(mask_length, 1)
    if max_targets == 0:
        return torch.zeros(B, 0, dtype=torch.bool, device=device)

    n_spans = (mask_prob * target_lengths.float() / mask_length).round().long().clamp(min=1)
    max_spans = int(n_spans.max().item())
    max_starts = (target_lengths - mask_length + 1).clamp(min=1)

    # Sample start positions: (B, max_spans)
    starts = (torch.rand(B, max_spans) * max_starts.float().unsqueeze(1)).long()

    # Expand each span to mask_length positions: (B, max_spans * mask_length)
    offsets = torch.arange(mask_length).view(1, 1, mask_length)
    positions = (starts.unsqueeze(2) + offsets).reshape(B, max_spans * mask_length)

    # Valid = span index < n_spans AND position < sequence length
    span_valid = (torch.arange(max_spans).unsqueeze(0) < n_spans.unsqueeze(1))  # (B, max_spans)
    span_valid = span_valid.unsqueeze(2).expand(-1, -1, mask_length).reshape(B, max_spans * mask_length)
    pos_valid = positions < target_lengths.unsqueeze(1)
    valid = span_valid & pos_valid

    positions_clamped = positions.clamp(0, max_targets - 1)
    mask = torch.zeros(B, max_targets, dtype=torch.bool)
    mask.scatter_(1, positions_clamped, valid)
    return mask.to(device)


def corrupt_mel_from_target_mask(mel, masked_positions, chunk_size, chunk_stride, mask_value):
    """Mask the full chunk_stride window of mel frames per masked target.

    A masked target at position t covers mel frames [t*stride, t*stride+stride),
    not [t*stride, t*stride+chunk_size). When chunk_size < chunk_stride, masking
    only the chunk_size frames leaves mel frames in the same encoder-downsample
    window unmasked, leaking context to the encoder.
    """
    bsz, num_targets = masked_positions.shape
    time_steps = mel.size(1)
    corrupted = mel.clone()
    masked = masked_positions.nonzero(as_tuple=False)
    if masked.numel() == 0:
        return corrupted
    span = max(chunk_stride, chunk_size)
    offsets = torch.arange(span, device=mel.device)
    frames = masked[:, 1:2] * chunk_stride + offsets[None, :]
    valid = frames < time_steps
    batch_idx = masked[:, 0:1].expand_as(frames)
    frame_mask = torch.zeros(bsz, time_steps, dtype=torch.bool, device=mel.device)
    frame_mask[batch_idx[valid], frames[valid]] = True
    corrupted[frame_mask] = mask_value
    return corrupted


class BucketDistributedSampler(Sampler):
    """
    Groups items by sequence length into buckets, then shuffles buckets and
    distributes across DDP ranks. Reduces padding waste on variable-length data.
    """

    def __init__(self, lengths, batch_size, num_replicas, rank, bucket_size_multiplier=100, seed=0, epoch=0):
        self.lengths = lengths
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.bucket_size = batch_size * bucket_size_multiplier
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.argsort(torch.tensor(self.lengths)).tolist()
        # Split sorted indices into buckets and shuffle within each bucket
        buckets = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        for bucket in buckets:
            perm = torch.randperm(len(bucket), generator=g).tolist()
            bucket[:] = [bucket[p] for p in perm]
        # Shuffle bucket order
        bucket_order = torch.randperm(len(buckets), generator=g).tolist()
        flat = [idx for b in bucket_order for idx in buckets[b]]
        # Trim to divisible by num_replicas
        total = (len(flat) // self.num_replicas) * self.num_replicas
        flat = flat[:total]
        # Assign this rank's slice
        flat = flat[self.rank:total:self.num_replicas]
        return iter(flat)

    def __len__(self):
        return len(self.lengths) // self.num_replicas


def dataloader_worker_init(_worker_id):
    torch.set_num_threads(1)


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    os.makedirs(args.output_dir, exist_ok=True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    trace(args.trace_startup, rank, "validating target metadata")
    metadata = validate_target_metadata(args.targets_dir, args)
    trace(args.trace_startup, rank, "target metadata validated")

    trace(args.trace_startup, rank, "building dataset; loading targets on each rank")
    dataset = SpecUnitDataset(
        data_root=args.data_root,
        splits=args.splits if args.splits else TRAIN_SPLITS,
        targets_path=os.path.join(args.targets_dir, "targets.pt"),
        cmvn_path=os.path.join(args.targets_dir, "cmvn.pt"),
        mel_cache_dir=args.mel_cache_dir,
    )
    trace(args.trace_startup, rank, f"dataset built with {len(dataset)} items")
    if args.bucket_sampler:
        lengths = [dataset.targets[it["uid"]]["z100"].numel() for it in dataset.items]
        sampler = BucketDistributedSampler(
            lengths=lengths,
            batch_size=args.batch_size,
            num_replicas=world_size,
            rank=rank,
        )
    elif world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None
    worker_kwargs = {}
    if args.workers > 0:
        worker_kwargs = {
            "persistent_workers": True,
            "prefetch_factor": max(1, args.prefetch_factor),
        }
    dataloader_timeout = args.dataloader_timeout if args.workers > 0 else 0
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_ssl,
        pin_memory=True,
        drop_last=True,
        timeout=dataloader_timeout,
        **worker_kwargs,
    )
    trace(args.trace_startup, rank, f"dataloader built with {len(loader)} batches")

    model = CausalSpecUnitSSL(variant=args.variant).to(device)
    if args.compile:
        try:
            import triton  # noqa: F401
            model = torch.compile(model)
            trace(args.trace_startup, rank, "model compiled with torch.compile")
        except ImportError:
            print0(rank, "[ssl warn] triton not available; skipping torch.compile")
            args.compile = False
    trace(args.trace_startup, rank, "model moved to device")
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        trace(args.trace_startup, rank, "DDP wrapper built")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / max(1, args.grad_accum_steps))
    scheduler = build_extended_noam_scheduler(
        optimizer,
        steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        peak_epochs=args.peak_epochs,
        decay_rate=args.noam_decay_rate,
    )
    optimizer_steps = 0
    start_epoch = 1

    if args.resume:
        trace(args.trace_startup, rank, f"loading checkpoint from {args.resume}")
        ckpt = load_checkpoint(args.resume, unwrap_model(model), optimizer, scheduler, device=device)
        optimizer_steps = int(ckpt.get("optimizer_steps", 0))
        start_epoch = int(ckpt.get("epoch", 1)) + 1
        print0(rank, f"[ssl] resumed from {args.resume} | opt_step={optimizer_steps} start_epoch={start_epoch}")

    print0(
        rank,
        f"CausalSpecUnit SSL | train={len(dataset)} utt | world={world_size} | "
        f"effective_batch={args.batch_size * world_size * args.grad_accum_steps} | "
        f"chunk={metadata['chunk_size']} stride={metadata['chunk_stride']} | "
        f"lr={args.lr:g} warmup={args.warmup_epochs} hold={args.peak_epochs}",
    )
    if args.max_steps is not None:
        print0(rank, f"[ssl] target={args.max_steps} steps | remaining={max(0, args.max_steps - optimizer_steps)}")

    step_times = []
    job_start = time.time()

    def fmt_eta(remaining_steps):
        if remaining_steps is None or remaining_steps <= 0:
            return "?"
        elapsed = time.time() - job_start
        if optimizer_steps <= 0 or elapsed <= 0:
            return "?"
        sps = optimizer_steps / elapsed
        eta_sec = remaining_steps / sps
        h, m = divmod(int(eta_sec), 3600)
        m, s = divmod(m, 60)
        return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if args.max_steps is not None and optimizer_steps >= args.max_steps:
                break
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = total_c100 = total_c500 = total_masked_frac = 0.0
            n_batches = 0
            skipped_steps = 0
            show = args.progress == "on" and is_main_process(rank)
            bar = tqdm(loader, desc=f"SSL {epoch:03d}", leave=False, disable=not show)

            for step, batch in enumerate(bar, start=1):
                if args.trace_every > 0 and step % args.trace_every == 0:
                    trace(True, rank, f"epoch={epoch} step={step} batch received")
                if args.max_train_batches is not None and step > args.max_train_batches:
                    break
                mel, lengths, z100, z500, target_lengths = batch
                mel = mel.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                z100 = z100.to(device, non_blocking=True)
                z500 = z500.to(device, non_blocking=True)

                masked_positions = make_hubert_mask(
                    target_lengths=target_lengths,
                    max_targets=z100.size(1),
                    mask_prob=args.mask_prob,
                    mask_length=args.mask_length,
                    device=device,
                )
                target_lengths = target_lengths.to(device, non_blocking=True)
                corrupted_mel = corrupt_mel_from_target_mask(
                    mel=mel,
                    masked_positions=masked_positions,
                    chunk_size=args.chunk_size,
                    chunk_stride=args.chunk_stride,
                    mask_value=args.mask_value,
                )

                sync_step = step % max(1, args.grad_accum_steps) == 0 or step == len(loader)
                window_start = ((step - 1) // max(1, args.grad_accum_steps)) * max(1, args.grad_accum_steps) + 1
                window_end = min(window_start + max(1, args.grad_accum_steps) - 1, len(loader))
                actual_accum_steps = window_end - window_start + 1
                sync_context = model.no_sync if isinstance(model, DDP) and not sync_step else contextlib.nullcontext

                with sync_context():
                    if args.trace_every > 0 and step % args.trace_every == 0:
                        trace(True, rank, f"epoch={epoch} step={step} forward start")
                    coarse_logits, fine_logits, _ = model(corrupted_mel, lengths)
                    coarse_logits, fine_logits, z100_aligned, z500_aligned, masked_aligned = align_ssl_tensors(
                        coarse_logits,
                        fine_logits,
                        z100,
                        z500,
                        masked_positions,
                    )
                    loss100 = masked_unit_ce(coarse_logits, z100_aligned, masked_aligned)
                    loss500 = masked_unit_ce(fine_logits, z500_aligned, masked_aligned)
                    loss = (loss100 + loss500) / actual_accum_steps
                    if args.trace_every > 0 and step % args.trace_every == 0:
                        trace(True, rank, f"epoch={epoch} step={step} backward start")
                    loss.backward()

                if sync_step:
                    if args.trace_every > 0 and step % args.trace_every == 0:
                        trace(True, rank, f"epoch={epoch} step={step} optimizer start")
                    t0 = time.time()
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    grad_norm_value = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    grad_is_safe = (
                        math.isfinite(grad_norm_value)
                        and (args.max_safe_grad_norm <= 0.0 or grad_norm_value <= args.max_safe_grad_norm)
                    )
                    if grad_is_safe:
                        optimizer.step()
                        scheduler.step()
                        optimizer_steps += 1
                        step_times.append(time.time() - t0)
                        if len(step_times) > 200:
                            step_times.pop(0)
                    else:
                        skipped_steps += 1
                        print0(
                            rank,
                            f"[ssl warn] unsafe grad norm {grad_norm_value:.2f} at epoch {epoch}, "
                            f"batch {step}; skipping optimizer and scheduler step",
                        )
                    optimizer.zero_grad(set_to_none=True)

                loss_val = (loss100 + loss500).detach().float().item()
                total_loss += loss_val
                total_c100 += loss100.detach().float().item()
                total_c500 += loss500.detach().float().item()
                total_masked_frac += masked_aligned.float().mean().detach().item()
                n_batches += 1
                if show:
                    remaining = (args.max_steps - optimizer_steps) if args.max_steps else None
                    postfix = dict(
                        loss=f"{loss_val:.3f}",
                        c100=f"{loss100.item():.3f}",
                        c500=f"{loss500.item():.3f}",
                        mask=f"{masked_aligned.float().mean().item():.2%}",
                        lr=f"{scheduler.get_last_lr()[0]:.1e}",
                        step=f"{optimizer_steps}/{args.max_steps or '?'}",
                        eta=fmt_eta(remaining) if remaining is not None else "?",
                    )
                    bar.set_postfix(**postfix, refresh=False)
                if args.log_every > 0 and is_main_process(rank) and step % args.log_every == 0:
                    remaining = (args.max_steps - optimizer_steps) if args.max_steps else None
                    elapsed = time.time() - job_start
                    tqdm.write(
                        f"[ssl] epoch={epoch} batch={step}/{len(loader)} opt_step={optimizer_steps} "
                        f"loss={loss_val:.4f} elapsed={elapsed/3600:.2f}h eta={fmt_eta(remaining)}"
                    )
                if args.max_steps is not None and optimizer_steps >= args.max_steps:
                    break

            avg = total_loss / max(n_batches, 1)
            elapsed = time.time() - job_start
            remaining = (args.max_steps - optimizer_steps) if args.max_steps else None
            skipped_note = f" skipped={skipped_steps}" if skipped_steps else ""
            print0(
                rank,
                f"[ssl] epoch={epoch:03d} opt_step={optimizer_steps} loss={avg:.4f} "
                f"c100={total_c100/max(n_batches,1):.4f} c500={total_c500/max(n_batches,1):.4f} "
                f"masked={total_masked_frac/max(n_batches,1):.2%}{skipped_note} "
                f"elapsed={elapsed/3600:.2f}h eta={fmt_eta(remaining)}",
            )
            if total_masked_frac / max(n_batches, 1) < 0.01 and is_main_process(rank):
                print(f"[ssl warn] epoch {epoch}: average masked fraction is <1% — check mask_prob/mask_length settings", flush=True)
            trace(args.trace_startup or args.trace_every > 0, rank, f"epoch={epoch} entering barrier")
            barrier()
            if is_main_process(rank) and (epoch % args.save_every == 0 or (args.max_steps is not None and optimizer_steps >= args.max_steps)):
                trace(args.trace_startup or args.trace_every > 0, rank, f"epoch={epoch} saving checkpoint step={optimizer_steps}")
                save_checkpoint(
                    os.path.join(args.output_dir, f"checkpoint_step{optimizer_steps:06d}"),
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    extra={"ssl_loss": avg, "optimizer_steps": optimizer_steps},
                )
                cleanup_checkpoints(args.output_dir, args.keep_checkpoints)
            trace(args.trace_startup or args.trace_every > 0, rank, f"epoch={epoch} leaving checkpoint barrier")
            barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main = record(main)
    main()
