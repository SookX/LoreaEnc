"""Standalone CTC fine-tuning for the MelHuBERT-Transformer (mh9m) encoder.

This file is intentionally independent of train_ctc.py: it implements only
SSL-checkpoint-init + CTC fine-tuning, with the same hyperparameters as the
SqueezeFormer-XS recipe in 10_benchmark_1h_10h_100h_3seeds.sh, but the model
class is the MelHuBERT-Transformer encoder plus a single Linear CTC head.

Why standalone: the shared train_ctc.py code path was hanging under DDP for
this encoder, with rank desync at end-of-epoch. The cleanest debug surface
is a minimal script that does exactly one thing.

Submit through 16_melhubert_finetune_standalone.sh, or invoke directly:
    torchrun --nproc_per_node=4 -m CausalSpecUnit.finetune_mh9m \
        --data-root dataset/datasets/librispeech/LibriSpeech \
        --cmvn-path outputs/causal_specunit/targets_960h_c8/cmvn.pt \
        --tokenizer-path dataset/bpe128.model \
        --ssl-checkpoint outputs/causal_specunit/melhubert_transformer_mh9m/ssl_fine_150000/checkpoint_step150000/checkpoint.pt \
        --train-split librilight_10h --batch-size 32 --epochs 150 \
        --output-dir outputs/causal_specunit/mh9m_ft/librilight_10h_seed42 --seed 42
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from SqueezeFormer.train import build_tokenizer
from CausalSpecUnit.common import build_lpft_scheduler
from CausalSpecUnit.data import CTCSpecDataset, collate_eval
from CausalSpecUnit.model import (
    MELHUBERT_TRANSFORMER_CONFIGS,
    MelHuBERTTransformerEncoder,
)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, msg: str):
    if is_main(rank):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Model: MelHuBERT encoder + linear CTC head
# ---------------------------------------------------------------------------

class MH9MCTCModel(nn.Module):
    def __init__(self, vocab_size: int, variant: str = "mh9m"):
        super().__init__()
        cfg = MELHUBERT_TRANSFORMER_CONFIGS[variant]
        self.encoder = MelHuBERTTransformerEncoder(cfg, layer_drop_p=0.0)
        self.head = nn.Linear(cfg.encoder_dim, vocab_size)

    def forward(self, mel: torch.Tensor, lengths: torch.Tensor):
        # mel: (B, T, n_mels), lengths: (B,) in mel frames.
        out, out_lengths, _ = self.encoder(mel, lengths, intermediate_layers=None)
        logits = self.head(out)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, out_lengths


def load_ssl_encoder(model: MH9MCTCModel, checkpoint_path: str, rank: int):
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = sd.get("model", sd)  # tolerate either format
    # Strip a leading "module." (DDP) and a leading "encoder." prefix so we
    # can load into our `.encoder` submodule with strict=False.
    cleaned = {}
    for k, v in state.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]
        if kk.startswith("encoder."):
            kk = kk[len("encoder."):]
        cleaned[kk] = v
    missing, unexpected = model.encoder.load_state_dict(cleaned, strict=False)
    log(rank, f"  SSL encoder loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        log(rank, f"    missing (first 3): {missing[:3]}")
    if unexpected[:3]:
        log(rank, f"    unexpected (first 3): {unexpected[:3]}")


# ---------------------------------------------------------------------------
# Scheduler: Noam-style warmup + inverse-sqrt decay (matches script 10)
# ---------------------------------------------------------------------------

def make_noam_scheduler(optimizer, warmup_steps: int, peak_steps: int, decay_rate: float = 0.5):
    """LR multiplier:
       step <= warmup_steps             -> step / warmup_steps
       warmup_steps < step <= peak       -> 1.0 (constant)
       step > peak                       -> (peak/step)^decay_rate
    Applied per parameter group via the same lambda.
    """
    def lr_lambda(step: int) -> float:
        if step <= warmup_steps:
            return float(step) / max(1, warmup_steps)
        if step <= peak_steps:
            return 1.0
        return (float(peak_steps) / float(step)) ** decay_rate
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# SpecAugment (time + frequency masking, masking with zeros)
# ---------------------------------------------------------------------------

class SpecAug(nn.Module):
    def __init__(self, time_mask_param: int, freq_mask_param: int,
                 time_masks: int, freq_masks: int):
        super().__init__()
        self.tmp, self.fmp = time_mask_param, freq_mask_param
        self.tm, self.fm = time_masks, freq_masks

    @torch.no_grad()
    def forward(self, mel: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mel
        B, T, F_ = mel.shape
        device = mel.device
        for _ in range(self.tm):
            t = torch.randint(0, max(1, self.tmp + 1), (B,), device=device)
            t0 = (torch.rand(B, device=device) * (lengths.float() - t.float()).clamp(min=1)).long()
            for b in range(B):
                if t[b] > 0:
                    mel[b, t0[b]:t0[b] + t[b], :] = 0.0
        for _ in range(self.fm):
            f = torch.randint(0, max(1, self.fmp + 1), (B,), device=device)
            f0 = (torch.rand(B, device=device) * (F_ - f.float()).clamp(min=1)).long()
            for b in range(B):
                if f[b] > 0:
                    mel[b, :, f0[b]:f0[b] + f[b]] = 0.0
        return mel


# ---------------------------------------------------------------------------
# CTC greedy decode + WER
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int):
    """log_probs: (B, T, V). Returns list of list-of-token-ids per item."""
    preds = log_probs.argmax(dim=-1)  # (B, T)
    out = []
    for b in range(preds.size(0)):
        seq = preds[b, : lengths[b].item()].tolist()
        # Collapse repeats then strip blanks (standard CTC collapse).
        collapsed, prev = [], None
        for x in seq:
            if x != prev:
                collapsed.append(x)
            prev = x
        out.append([x for x in collapsed if x != blank_id])
    return out


def word_error_counts(ref: list, hyp: list):
    """Substitutions, insertions, deletions via Levenshtein."""
    R, H = len(ref), len(hyp)
    if R == 0:
        return 0, H, 0
    D = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        D[i][0] = i
    for j in range(H + 1):
        D[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            if ref[i - 1] == hyp[j - 1]:
                D[i][j] = D[i - 1][j - 1]
            else:
                D[i][j] = 1 + min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
    # Backtrace counts.
    sub = ins = dele = 0
    i, j = R, H
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1]:
            i -= 1; j -= 1
        elif D[i][j] == D[i - 1][j - 1] + 1:
            sub += 1; i -= 1; j -= 1
        elif D[i][j] == D[i - 1][j] + 1:
            dele += 1; i -= 1
        else:
            ins += 1; j -= 1
    dele += i; ins += j
    return sub, ins, dele


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, tokenizer, blank_id, ctc, device, rank, world_size):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_sub = total_ins = total_del = 0
    total_ref_words = 0
    example = None
    for mel, lengths, labels, label_lengths, transcripts in loader:
        mel = mel.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            log_probs, out_lengths = model(mel, lengths)
            loss = ctc(log_probs.transpose(0, 1), labels, out_lengths, label_lengths)
        total_loss += float(loss.detach().item())
        total_batches += 1
        hyp_ids = greedy_decode(log_probs, out_lengths, blank_id)
        for ids, ref_text in zip(hyp_ids, transcripts):
            hyp_text = tokenizer.decode(ids).lower().strip()
            ref_words = ref_text.lower().strip().split()
            hyp_words = hyp_text.split()
            s, i_, d = word_error_counts(ref_words, hyp_words)
            total_sub += s; total_ins += i_; total_del += d
            total_ref_words += max(len(ref_words), 1)
            if example is None:
                example = (hyp_text, ref_text)
    stats = torch.tensor(
        [total_loss, total_batches, total_sub, total_ins, total_del, total_ref_words],
        dtype=torch.float64, device=device,
    )
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_avg = float(stats[0]) / max(float(stats[1]), 1.0)
    wer = float(stats[2] + stats[3] + stats[4]) / max(float(stats[5]), 1.0)
    return {
        "loss": loss_avg,
        "wer": wer,
        "del_rate": float(stats[4]) / max(float(stats[5]), 1.0),
        "ins_rate": float(stats[3]) / max(float(stats[5]), 1.0),
        "sub_rate": float(stats[2]) / max(float(stats[5]), 1.0),
        "example": example,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--cmvn-path", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--ssl-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-split", default="librilight_10h")
    p.add_argument("--dev-split", default="dev-other")
    p.add_argument("--variant", default="mh9m")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--encoder-lr", type=float, default=3e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--peak-epochs", type=int, default=50)
    p.add_argument("--noam-decay-rate", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    # LP-FT: keep the encoder update-frozen (LR=0) for N epochs so the CTC head
    # adapts first, then linearly re-warmup. Default 0/0 == the original recipe.
    p.add_argument("--freeze-encoder-epochs", type=int, default=0)
    p.add_argument("--encoder-rewarmup-epochs", type=int, default=0)
    p.add_argument("--specaug-time-mask-param", type=int, default=30)
    p.add_argument("--specaug-freq-mask-param", type=int, default=20)
    p.add_argument("--specaug-time-masks", type=int, default=2)
    p.add_argument("--specaug-freq-masks", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rank, local_rank, world_size = setup_dist()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    if is_main(rank):
        os.makedirs(args.output_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    log(rank, f"world_size={world_size} rank={rank} local_rank={local_rank} device={device}")
    log(rank, f"train={args.train_split} epochs={args.epochs} batch={args.batch_size} workers={args.workers}")

    tokenizer = build_tokenizer(args.tokenizer_path)
    vocab_size = tokenizer.vocab_size
    blank_id = vocab_size  # blank is the EXTRA token beyond the BPE vocab

    log(rank, f"tokenizer vocab={vocab_size} blank_id={blank_id}")

    train_ds = CTCSpecDataset(args.data_root, [args.train_split], tokenizer,
                              cmvn_path=args.cmvn_path, train_split=True)
    dev_ds = CTCSpecDataset(args.data_root, [args.dev_split], tokenizer,
                            cmvn_path=args.cmvn_path, train_split=False,
                            validate_audio=True)
    log(rank, f"train_n={len(train_ds)} dev_n={len(dev_ds)}")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                        shuffle=True, drop_last=True, seed=args.seed) if world_size > 1 else None
    dev_sampler = DistributedSampler(dev_ds, num_replicas=world_size, rank=rank,
                                      shuffle=False, drop_last=False, seed=args.seed) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=args.workers,
        collate_fn=collate_eval, pin_memory=True, drop_last=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=4 if args.workers > 0 else None,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.eval_batch_size, sampler=dev_sampler,
        shuffle=False, num_workers=args.workers,
        collate_fn=collate_eval, pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=4 if args.workers > 0 else None,
    )

    model = MH9MCTCModel(vocab_size=blank_id + 1, variant=args.variant).to(device)
    load_ssl_encoder(model, args.ssl_checkpoint, rank)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    enc_params = [p for n, p in (model.module if world_size > 1 else model).named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in (model.module if world_size > 1 else model).named_parameters() if n.startswith("head.")]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.encoder_lr, "name": "encoder"},
         {"params": head_params, "lr": args.head_lr, "name": "head"}],
        weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-6,
    )

    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = args.warmup_epochs * steps_per_epoch
    peak_steps = args.peak_epochs * steps_per_epoch
    if args.freeze_encoder_epochs > 0 or args.encoder_rewarmup_epochs > 0:
        # Encoder group (name "encoder") stays at LR 0 during freeze, then
        # re-warms; head group is unaffected. DDP-safe: params keep requires_grad
        # (grads still all-reduce), only the LR is driven to 0.
        scheduler = build_lpft_scheduler(
            optimizer, steps_per_epoch,
            warmup_epochs=args.warmup_epochs, peak_epochs=args.peak_epochs,
            decay_rate=args.noam_decay_rate,
            encoder_freeze_epochs=args.freeze_encoder_epochs,
            encoder_rewarmup_epochs=args.encoder_rewarmup_epochs,
        )
        log(rank, f"LP-FT: encoder freeze={args.freeze_encoder_epochs}ep rewarmup={args.encoder_rewarmup_epochs}ep")
    else:
        scheduler = make_noam_scheduler(optimizer, warmup_steps, peak_steps, args.noam_decay_rate)

    specaug = SpecAug(args.specaug_time_mask_param, args.specaug_freq_mask_param,
                      args.specaug_time_masks, args.specaug_freq_masks)

    ctc = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    n_params = sum(p.numel() for p in model.parameters())
    log(rank, f"model params={n_params/1e6:.2f}M (vocab={vocab_size} + blank)")

    best_wer = float("inf")
    metrics_path = os.path.join(args.output_dir, "ctc_metrics.jsonl")
    if is_main(rank):
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"event": "run_start", "world_size": world_size,
                                 "args": vars(args), "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")

    global_step = 0
    run_start = time.time()
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        specaug.train()
        epoch_loss = 0.0
        n_batches = 0
        t_epoch = time.time()
        for batch in train_loader:
            mel, lengths, labels, label_lengths, _ = batch
            mel = mel.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            label_lengths = label_lengths.to(device, non_blocking=True)
            mel = specaug(mel, lengths)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                log_probs, out_lengths = model(mel, lengths)
                loss = ctc(log_probs.transpose(0, 1), labels, out_lengths, label_lengths)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += float(loss.detach().item())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_secs = time.time() - t_epoch
        if world_size > 1:
            avg = torch.tensor([avg_loss, n_batches], dtype=torch.float64, device=device)
            dist.all_reduce(avg, op=dist.ReduceOp.SUM)
            avg_loss = float(avg[0]) / max(float(avg[1]), 1.0) * world_size / max(world_size, 1)

        m = evaluate(model, dev_loader, tokenizer, blank_id, ctc, device, rank, world_size)
        new_best = m["wer"] < best_wer
        if new_best:
            best_wer = m["wer"]

        log(rank,
            f"epoch={epoch:03d} train_loss={avg_loss:.4f} dev_loss={m['loss']:.4f} "
            f"wer={100*m['wer']:.2f}% del={100*m['del_rate']:.2f}% ins={100*m['ins_rate']:.2f}% sub={100*m['sub_rate']:.2f}% "
            f"best={100*best_wer:.2f}% epoch_secs={epoch_secs:.1f} lr={scheduler.get_last_lr()[0]:.2e}")
        if is_main(rank) and m.get("example") is not None:
            hyp, ref = m["example"]
            log(rank, f"  REF: {ref}")
            log(rank, f"  HYP: {hyp}")
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"event": "epoch_end", "epoch": epoch,
                                     "train_loss": avg_loss, "dev_loss": m["loss"],
                                     "wer": m["wer"], "best_wer": best_wer,
                                     "del_rate": m["del_rate"], "ins_rate": m["ins_rate"],
                                     "sub_rate": m["sub_rate"], "elapsed_hours": (time.time() - run_start) / 3600}) + "\n")
            if new_best:
                ckpt_dir = os.path.join(args.output_dir, "checkpoint_best")
                os.makedirs(ckpt_dir, exist_ok=True)
                state = (model.module if world_size > 1 else model).state_dict()
                torch.save({"model": state, "epoch": epoch, "wer": best_wer,
                             "vocab_size": vocab_size, "blank_id": blank_id,
                             "variant": args.variant},
                            os.path.join(ckpt_dir, "checkpoint.pt"))

    log(rank, f"DONE. best_wer={100*best_wer:.2f}%")
    cleanup_dist()


if __name__ == "__main__":
    main()
