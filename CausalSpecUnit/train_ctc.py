import argparse
import json
import math
import os
import shutil
import socket
import sys
import time

import torch
import torch.distributed as dist
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


def current_group_grad_norms(optimizer):
    norms = {}
    for idx, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group_{idx}")
        total_sq = 0.0
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            total_sq += float(grad.pow(2).sum().item())
        norms[name] = math.sqrt(total_sq)
    return norms


def reduce_train_average(total_loss, n_batches, device):
    stats = torch.tensor([float(total_loss), float(n_batches)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item() / max(stats[1].item(), 1.0))


class BatchSpecAugment(nn.Module):
    """SpecAugment for padded [B, T, F] CMVN log-mel batches."""

    def __init__(
        self,
        time_mask_param=40,
        freq_mask_param=30,
        num_time_masks=2,
        num_freq_masks=2,
        mask_value=0.0,
    ):
        super().__init__()
        self.time_mask_param = int(time_mask_param)
        self.freq_mask_param = int(freq_mask_param)
        self.num_time_masks = int(num_time_masks)
        self.num_freq_masks = int(num_freq_masks)
        self.mask_value = float(mask_value)

    def forward(self, mel, lengths):
        if self.num_time_masks <= 0 and self.num_freq_masks <= 0:
            return mel
        out = mel.clone()
        batch, _, n_mels = out.shape
        device = out.device

        for b in range(batch):
            valid_t = int(lengths[b].item())
            if valid_t <= 0:
                continue
            for _ in range(self.num_freq_masks):
                width_max = min(self.freq_mask_param, n_mels)
                if width_max <= 0:
                    continue
                width = int(torch.randint(0, width_max + 1, (1,), device=device).item())
                if width == 0 or width >= n_mels:
                    continue
                start = int(torch.randint(0, n_mels - width + 1, (1,), device=device).item())
                out[b, :valid_t, start:start + width] = self.mask_value
            for _ in range(self.num_time_masks):
                width_max = min(self.time_mask_param, valid_t)
                if width_max <= 0:
                    continue
                width = int(torch.randint(0, width_max + 1, (1,), device=device).item())
                if width == 0 or width >= valid_t:
                    continue
                start = int(torch.randint(0, valid_t - width + 1, (1,), device=device).item())
                out[b, start:start + width, :] = self.mask_value
        return out


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
    p.add_argument("--specaug", action="store_true",
                   help="Apply SpecAugment to training mels after CMVN and before the model.")
    p.add_argument("--specaug-time-mask-param", type=int, default=40)
    p.add_argument("--specaug-freq-mask-param", type=int, default=30)
    p.add_argument("--specaug-time-masks", type=int, default=2)
    p.add_argument("--specaug-freq-masks", type=int, default=2)
    p.add_argument("--specaug-disable-last-epochs", type=int, default=0,
                   help="Disable SpecAugment for the final N epochs for clean fine-tuning.")
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


def word_error_counts(ref_words, hyp_words):
    """Return substitutions, insertions, deletions for ref -> hyp."""
    rows, cols = len(ref_words), len(hyp_words)
    dp = [[(0, 0, 0, 0) for _ in range(cols + 1)] for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        cost, sub, ins, dele = dp[i - 1][0]
        dp[i][0] = (cost + 1, sub, ins, dele + 1)
    for j in range(1, cols + 1):
        cost, sub, ins, dele = dp[0][j - 1]
        dp[0][j] = (cost + 1, sub, ins + 1, dele)

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            cost, sub, ins, dele = dp[i - 1][j - 1]
            substitute = (cost + 1, sub + 1, ins, dele)
            cost, sub, ins, dele = dp[i][j - 1]
            insert = (cost + 1, sub, ins + 1, dele)
            cost, sub, ins, dele = dp[i - 1][j]
            delete = (cost + 1, sub, ins, dele + 1)
            dp[i][j] = min((substitute, insert, delete), key=lambda x: (x[0], x[1], x[2], x[3]))
    _, sub, ins, dele = dp[rows][cols]
    return sub, ins, dele


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
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0
    total_hyp_words = 0
    total_char_errors = 0
    total_ref_chars = 0
    empty_hypotheses = 0
    total_utterances = 0
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
            substitutions, insertions, deletions = word_error_counts(rw, hw)
            total_substitutions += substitutions
            total_insertions += insertions
            total_deletions += deletions
            total_word_errors += substitutions + insertions + deletions
            total_ref_words += max(len(rw), 1)
            total_hyp_words += len(hw)
            empty_hypotheses += int(len(hw) == 0)
            ref_chars = list(ref.replace(" ", ""))
            hyp_chars = list(hyp.replace(" ", ""))
            total_char_errors += edit_distance(hyp_chars, ref_chars)
            total_ref_chars += max(len(ref_chars), 1)
            total_utterances += 1
            if example is None:
                example = (hyp, ref)

    stats = torch.tensor(
        [
            total_word_errors,
            total_ref_words,
            total_loss,
            total_batches,
            total_substitutions,
            total_insertions,
            total_deletions,
            total_hyp_words,
            total_char_errors,
            total_ref_chars,
            empty_hypotheses,
            total_utterances,
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    total_word_errors = float(stats[0].item())
    total_ref_words = float(stats[1].item())
    total_loss = float(stats[2].item())
    total_batches = float(stats[3].item())
    total_substitutions = float(stats[4].item())
    total_insertions = float(stats[5].item())
    total_deletions = float(stats[6].item())
    total_hyp_words = float(stats[7].item())
    total_char_errors = float(stats[8].item())
    total_ref_chars = float(stats[9].item())
    empty_hypotheses = float(stats[10].item())
    total_utterances = float(stats[11].item())

    return {
        "loss": total_loss / max(total_batches, 1.0),
        "wer": total_word_errors / max(total_ref_words, 1.0),
        "cer": total_char_errors / max(total_ref_chars, 1.0),
        "substitution_rate": total_substitutions / max(total_ref_words, 1.0),
        "insertion_rate": total_insertions / max(total_ref_words, 1.0),
        "deletion_rate": total_deletions / max(total_ref_words, 1.0),
        "hyp_words_per_ref_word": total_hyp_words / max(total_ref_words, 1.0),
        "empty_hypothesis_rate": empty_hypotheses / max(total_utterances, 1.0),
        "eval_utterances": int(total_utterances),
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
    specaugment = BatchSpecAugment(
        time_mask_param=args.specaug_time_mask_param,
        freq_mask_param=args.specaug_freq_mask_param,
        num_time_masks=args.specaug_time_masks,
        num_freq_masks=args.specaug_freq_masks,
    ).to(device)
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
        f"{hours_note} | warmup={args.warmup_epochs} hold={args.peak_epochs} decay={args.noam_decay_rate:g} "
        f"| specaug={args.specaug} disable_last={args.specaug_disable_last_epochs}",
    )

    try:
        for epoch in range(1, args.epochs + 1):
            specaug_enabled = bool(
                args.specaug
                and (
                    args.specaug_disable_last_epochs <= 0
                    or epoch <= args.epochs - args.specaug_disable_last_epochs
                )
            )
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            n_batches = 0
            grad_steps = 0
            clipped_steps = 0
            grad_norm_sum = 0.0
            grad_norm_max = 0.0
            group_grad_norm_sums = {}
            group_grad_norm_max = {}
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
                if specaug_enabled:
                    mel = specaugment(mel, lengths)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    log_probs, output_lengths = model(mel, lengths)
                    loss = ctc_loss(log_probs.transpose(0, 1), labels, output_lengths, label_lengths)
                    loss = loss / max(1, args.grad_accum_steps)
                loss.backward()
                sync_step = step % max(1, args.grad_accum_steps) == 0 or step == len(train_loader)
                grad_norm_value = None
                group_grad_norms = None
                if sync_step:
                    group_grad_norms = current_group_grad_norms(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    grad_norm_value = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    grad_steps += 1
                    grad_norm_sum += grad_norm_value
                    grad_norm_max = max(grad_norm_max, grad_norm_value)
                    clipped_steps += int(grad_norm_value > args.max_grad_norm)
                    for name, value in group_grad_norms.items():
                        group_grad_norm_sums[name] = group_grad_norm_sums.get(name, 0.0) + value
                        group_grad_norm_max[name] = max(group_grad_norm_max.get(name, 0.0), value)
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
                        "group_grad_norms": group_grad_norms,
                        "specaug_enabled": specaug_enabled,
                        "elapsed_hours": (time.time() - run_start) / 3600,
                    })

            avg_loss = reduce_train_average(total_loss, n_batches, device)
            grad_norm_avg = grad_norm_sum / max(grad_steps, 1)
            group_grad_norm_avg = {
                name: value / max(grad_steps, 1)
                for name, value in group_grad_norm_sums.items()
            }
            clip_fraction = clipped_steps / max(grad_steps, 1)
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
                        f"wer={metrics['wer']:.2%} cer={metrics['cer']:.2%} best={best_wer:.2%} "
                        f"del={metrics['deletion_rate']:.2%} ins={metrics['insertion_rate']:.2%} "
                        f"clip={clip_fraction:.1%}\nREF: {ref}\nHYP: {hyp}"
                    )
                    append_jsonl(metrics_path, {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "optimizer_step": optimizer_steps,
                        "train_loss": avg_loss,
                        "dev_loss": metrics["loss"],
                        "wer": metrics["wer"],
                        "cer": metrics["cer"],
                        "substitution_rate": metrics["substitution_rate"],
                        "insertion_rate": metrics["insertion_rate"],
                        "deletion_rate": metrics["deletion_rate"],
                        "hyp_words_per_ref_word": metrics["hyp_words_per_ref_word"],
                        "empty_hypothesis_rate": metrics["empty_hypothesis_rate"],
                        "eval_utterances": metrics["eval_utterances"],
                        "best_wer": best_wer,
                        "lr": scheduler.get_last_lr()[0],
                        "lrs": current_lrs(optimizer),
                        "specaug_enabled": specaug_enabled,
                        "specaug": {
                            "time_mask_param": args.specaug_time_mask_param,
                            "freq_mask_param": args.specaug_freq_mask_param,
                            "time_masks": args.specaug_time_masks,
                            "freq_masks": args.specaug_freq_masks,
                            "disable_last_epochs": args.specaug_disable_last_epochs,
                        },
                        "grad_norm_avg": grad_norm_avg,
                        "grad_norm_max": grad_norm_max,
                        "clip_fraction": clip_fraction,
                        "group_grad_norm_avg": group_grad_norm_avg,
                        "group_grad_norm_max": group_grad_norm_max,
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
                        "specaug_enabled": specaug_enabled,
                        "grad_norm_avg": grad_norm_avg,
                        "grad_norm_max": grad_norm_max,
                        "clip_fraction": clip_fraction,
                        "group_grad_norm_avg": group_grad_norm_avg,
                        "group_grad_norm_max": group_grad_norm_max,
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
