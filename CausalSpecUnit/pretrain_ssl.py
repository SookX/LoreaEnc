import argparse
import json
import math
import os
import shutil

import torch
import torch.nn as nn
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
    print0,
    save_checkpoint,
    setup_distributed,
)
from CausalSpecUnit.data import SpecUnitDataset, collate_ssl
from CausalSpecUnit.model import CausalSpecUnitSSL


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="dataset/datasets/librispeech/LibriSpeech")
    p.add_argument("--targets-dir", type=str, default="outputs/causal_specunit/targets")
    p.add_argument("--output-dir", type=str, default="outputs/causal_specunit/pretrain")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dataloader-timeout", type=int, default=120)
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
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--keep-checkpoints", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Stop after this many optimizer steps.")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--progress", choices=["on", "off"], default="on")
    return p.parse_args()


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


def make_hubert_mask(target_lengths, max_targets, mask_prob, mask_length, device):
    """Return [B, T_targets] bool mask over target positions."""
    mask = torch.zeros(target_lengths.size(0), max_targets, dtype=torch.bool, device=device)
    for b, length in enumerate(target_lengths.tolist()):
        if length <= 0:
            continue
        n_spans = max(1, int(round(mask_prob * length / max(mask_length, 1))))
        max_start = max(1, length - mask_length + 1)
        starts = torch.randint(0, max_start, (n_spans,), device=device)
        for start in starts.tolist():
            end = min(length, start + mask_length)
            mask[b, start:end] = True
    return mask


def corrupt_mel_from_target_mask(mel, masked_positions, chunk_size, chunk_stride, mask_value):
    """Mask clean spectrogram frames corresponding to masked target chunks."""
    corrupted = mel.clone()
    bsz, num_targets = masked_positions.shape
    time_steps = mel.size(1)
    for b in range(bsz):
        starts = torch.nonzero(masked_positions[b], as_tuple=False).flatten() * chunk_stride
        for start in starts.tolist():
            end = min(time_steps, start + chunk_size)
            corrupted[b, start:end, :] = mask_value
    return corrupted


def masked_unit_ce(logits, targets, masked_positions):
    """CE over only masked positions. Pads use target=-100 and are ignored."""
    t = min(logits.size(1), targets.size(1), masked_positions.size(1))
    logits = logits[:, :t]
    targets = targets[:, :t]
    masked_positions = masked_positions[:, :t] & targets.ne(-100)
    if not masked_positions.any():
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[masked_positions], targets[masked_positions])


def align_ssl_tensors(coarse_logits, fine_logits, z100, z500, masked_positions):
    """Crop model outputs, clean labels, and mask to one shared target length."""
    t = min(coarse_logits.size(1), fine_logits.size(1), z100.size(1), z500.size(1), masked_positions.size(1))
    return (
        coarse_logits[:, :t],
        fine_logits[:, :t],
        z100[:, :t],
        z500[:, :t],
        masked_positions[:, :t],
    )


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    os.makedirs(args.output_dir, exist_ok=True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    metadata = validate_target_metadata(args.targets_dir, args)

    dataset = SpecUnitDataset(
        data_root=args.data_root,
        splits=TRAIN_SPLITS,
        targets_path=os.path.join(args.targets_dir, "targets.pt"),
        cmvn_path=os.path.join(args.targets_dir, "cmvn.pt"),
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    worker_kwargs = {"persistent_workers": True, "prefetch_factor": 4} if args.workers > 0 else {}
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_ssl,
        pin_memory=True,
        drop_last=True,
        timeout=args.dataloader_timeout,
        **worker_kwargs,
    )

    model = CausalSpecUnitSSL(variant=args.variant).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / max(1, args.grad_accum_steps))
    scheduler = build_extended_noam_scheduler(optimizer, steps_per_epoch, warmup_epochs=20, peak_epochs=160, decay_rate=1.0)
    print0(
        rank,
        f"CausalSpecUnit SSL | train={len(dataset)} utt | world={world_size} | "
        f"effective_batch={args.batch_size * world_size * args.grad_accum_steps} | "
        f"chunk={metadata['chunk_size']} stride={metadata['chunk_stride']}",
    )
    optimizer_steps = 0

    try:
        for epoch in range(1, args.epochs + 1):
            if args.max_steps is not None and optimizer_steps >= args.max_steps:
                break
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = total_c100 = total_c500 = total_masked_frac = 0.0
            n_batches = 0
            show = args.progress == "on" and is_main_process(rank)
            bar = tqdm(loader, desc=f"SSL {epoch:03d}", leave=False, disable=not show)

            for step, batch in enumerate(bar, start=1):
                if args.max_train_batches is not None and step > args.max_train_batches:
                    break
                mel, lengths, z100, z500, target_lengths = batch
                mel = mel.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                z100 = z100.to(device, non_blocking=True)
                z500 = z500.to(device, non_blocking=True)
                target_lengths = target_lengths.to(device, non_blocking=True)

                masked_positions = make_hubert_mask(
                    target_lengths=target_lengths,
                    max_targets=z100.size(1),
                    mask_prob=args.mask_prob,
                    mask_length=args.mask_length,
                    device=device,
                )
                corrupted_mel = corrupt_mel_from_target_mask(
                    mel=mel,
                    masked_positions=masked_positions,
                    chunk_size=args.chunk_size,
                    chunk_stride=args.chunk_stride,
                    mask_value=args.mask_value,
                )

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
                loss = (loss100 + loss500) / max(1, args.grad_accum_steps)
                loss.backward()

                sync_step = step % max(1, args.grad_accum_steps) == 0 or step == len(loader)
                if sync_step:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1

                loss_val = (loss100 + loss500).detach().float().item()
                total_loss += loss_val
                total_c100 += loss100.detach().float().item()
                total_c500 += loss500.detach().float().item()
                total_masked_frac += masked_aligned.float().mean().detach().item()
                n_batches += 1
                if show:
                    bar.set_postfix(loss=f"{loss_val:.3f}", c100=f"{loss100.item():.3f}", c500=f"{loss500.item():.3f}", mask=f"{masked_aligned.float().mean().item():.2%}", lr=f"{scheduler.get_last_lr()[0]:.1e}", refresh=False)
                if args.log_every > 0 and is_main_process(rank) and step % args.log_every == 0:
                    tqdm.write(f"[ssl] epoch={epoch} batch={step}/{len(loader)} opt_step={optimizer_steps} loss={loss_val:.4f}")
                if args.max_steps is not None and optimizer_steps >= args.max_steps:
                    break

            avg = total_loss / max(n_batches, 1)
            print0(rank, f"[ssl] epoch={epoch:03d} opt_step={optimizer_steps} loss={avg:.4f} c100={total_c100/max(n_batches,1):.4f} c500={total_c500/max(n_batches,1):.4f} masked={total_masked_frac/max(n_batches,1):.2%}")
            barrier()
            if is_main_process(rank) and (epoch % args.save_every == 0 or (args.max_steps is not None and optimizer_steps >= args.max_steps)):
                save_checkpoint(
                    os.path.join(args.output_dir, f"checkpoint_step{optimizer_steps:06d}"),
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    extra={"ssl_loss": avg, "optimizer_steps": optimizer_steps},
                )
                cleanup_checkpoints(args.output_dir, args.keep_checkpoints)
            barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
