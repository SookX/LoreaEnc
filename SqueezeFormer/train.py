"""
SqueezeFormer XS — CTC training on LibriSpeech, plain PyTorch DDP.
Matches the paper (arXiv:2206.00888): BPE-128, 10ms mel hop, Noam LR,
effective batch 1024, 5-mask SpecAugment, weight-decay 5e-4.

Single GPU:
    python SqueezeFormer/train.py

Multi-GPU:
    torchrun --nproc_per_node=2 SqueezeFormer/train.py
"""

import os
import sys
import math
import random
import shutil
import argparse
import contextlib
import collections

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, Sampler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.dataset import LibriSpeechDataset, ConformerSpecAugment
from SqueezeFormer import Squeezeformer, get_config

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (paper: arXiv:2206.00888, XS variant)
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_SPLITS      = ["train-clean-100", "train-clean-360", "train-other-500"]
DEV_CLEAN_SPLIT   = "dev-clean"
DEV_OTHER_SPLIT   = "dev-other"
VARIANT           = "xs"

LEARNING_RATE     = 2e-3        # paper peak LR for XS
WEIGHT_DECAY      = 5e-4        # paper
ADAM_BETA1        = 0.9
ADAM_BETA2        = 0.98
ADAM_EPSILON      = 1e-8

NUM_EPOCHS        = 150
WARMUP_EPOCHS     = 20          # paper
PEAK_EPOCHS       = 160         # paper (LR held constant after warmup)

GRAD_ACCUM        = 2           # effective batch = 256/GPU × 2 GPU × 2 = 1024
SAVE_EVERY        = 1
NUM_KEEP_CHECKPOINTS = 5

MAX_MEL_FRAMES    = 2000        # 20s at 10ms hop → 4× subsample → 500 out frames
MEL_HOP           = 160         # 10ms at 16kHz — must match precompute_mels.py

BPE_MODEL         = "dataset/bpe128.model"


# ─────────────────────────────────────────────────────────────────────────────
# BPE TOKENIZER WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class BPETokenizer:
    def __init__(self, model_path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab_size   = self.sp.get_piece_size()   # 128
        self.pad_token_id = self.sp.pad_id()           # 0  ← CTC blank

    def encode(self, text):
        return self.sp.encode(text)          # List[int], no special tokens

    def decode(self, ids):
        return self.sp.decode(ids)           # str


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hours",      type=float, default=None)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--output-dir", type=str,   default="outputs/baseline/960h")
    p.add_argument("--run-name",   type=str,   default="baseline_960h")
    p.add_argument("--resume",     type=str,   default=None)
    p.add_argument("--data-root",  type=str,   default="dataset/datasets/librispeech/LibriSpeech")
    p.add_argument("--bpe-model",  type=str,   default=BPE_MODEL)
    p.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--workers",    type=int,   default=8)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DDP
# ─────────────────────────────────────────────────────────────────────────────

def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank, rank, world_size = 0, 0, 1
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def barrier(world_size):
    if world_size > 1:
        dist.barrier()


# ─────────────────────────────────────────────────────────────────────────────
# BUCKETED SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

class DistributedBucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, num_replicas, rank,
                 bucket_size_multiplier=100, shuffle=True, seed=0):
        self.lengths      = lengths
        self.batch_size   = batch_size
        self.num_replicas = num_replicas
        self.rank         = rank
        self.bucket_sz    = batch_size * bucket_size_multiplier
        self.shuffle      = shuffle
        self.seed         = seed
        self.epoch        = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        sorted_idx = torch.argsort(torch.tensor(self.lengths, dtype=torch.float32))
        buckets    = list(sorted_idx.split(self.bucket_sz))

        if self.shuffle:
            buckets = [b[torch.randperm(len(b), generator=g)] for b in buckets]
            order   = torch.randperm(len(buckets), generator=g).tolist()
            buckets = [buckets[i] for i in order]

        all_idx = torch.cat(buckets)
        n       = len(all_idx)
        effective    = self.batch_size * self.num_replicas
        total_global = math.ceil(n / effective)
        target_size  = total_global * effective
        if target_size > n:
            all_idx = torch.cat([all_idx, all_idx[: target_size - n]])

        all_batches = all_idx.view(-1, self.batch_size)
        my_batches  = all_batches[self.rank :: self.num_replicas]
        for batch in my_batches:
            yield batch.tolist()

    def __len__(self):
        return math.ceil(len(self.lengths) / (self.batch_size * self.num_replicas))


# ─────────────────────────────────────────────────────────────────────────────
# COLLATE
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    vals        = [item["input_values"][:MAX_MEL_FRAMES] for item in batch]
    label_list  = [item["labels"]                        for item in batch]

    lengths       = torch.tensor([v.size(0) for v in vals], dtype=torch.long)
    padded        = nn.utils.rnn.pad_sequence(vals, batch_first=True)
    label_lengths = torch.tensor([len(l) for l in label_list], dtype=torch.long)
    labels_flat   = torch.cat([
        l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in label_list
    ])
    return padded, lengths, labels_flat, label_lengths


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config, num_classes):
    return Squeezeformer(
        num_classes=num_classes,
        input_dim=config.input_dim,
        encoder_dim=config.encoder_dim,
        num_encoder_layers=config.num_encoder_layers,
        reduce_layer_index=config.reduce_layer_index,
        recover_layer_index=config.recover_layer_index,
        num_attention_heads=config.num_attention_heads,
        feed_forward_expansion_factor=config.feed_forward_expansion_factor,
        conv_expansion_factor=config.conv_expansion_factor,
        input_dropout_p=config.input_dropout_p,
        feed_forward_dropout_p=config.feed_forward_dropout_p,
        attention_dropout_p=config.attention_dropout_p,
        conv_dropout_p=config.conv_dropout_p,
        conv_kernel_size=config.conv_kernel_size,
        half_step_residual=config.half_step_residual,
        adaptive_scale=config.adaptive_scale,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LR SCHEDULE  (paper: warmup → constant peak → Noam decay)
# ─────────────────────────────────────────────────────────────────────────────

def build_noam_scheduler(optimizer, warmup_epochs, peak_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # linear warmup
        elif epoch < warmup_epochs + peak_epochs:
            return 1.0                                   # constant peak
        else:
            # Noam decay: 1/sqrt(decay_step)
            decay = epoch - warmup_epochs - peak_epochs + 1
            return ((warmup_epochs + peak_epochs) /
                    (warmup_epochs + peak_epochs + decay)) ** 0.5
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# DECODING / METRICS
# ─────────────────────────────────────────────────────────────────────────────

def greedy_decode(log_probs, lengths, blank_id):
    preds   = log_probs.argmax(dim=-1)
    decoded = []
    for b in range(preds.size(0)):
        seq, prev = [], None
        for tok in preds[b, :lengths[b]].tolist():
            if tok != prev:
                seq.append(tok)
            prev = tok
        decoded.append([t for t in seq if t != blank_id])
    return decoded


def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp   = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp  = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev  = temp
    return dp[n]


def word_error_rate(hyp, ref):
    return _edit_distance(hyp.lower().split(), ref.lower().split()) / max(len(ref.split()), 1)


# ─────────────────────────────────────────────────────────────────────────────
# DEV EVALUATION (rank-0 only)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dev(model_raw, dev_dataset, tokenizer, blank_id, device, batch_size=256):
    model_raw.eval()
    loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)
    total_wer, n = 0.0, 0
    examples      = []

    with torch.no_grad():
        for data, lengths, labels_flat, label_lengths in loader:
            mel     = data.to(device).float()
            lengths = lengths.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                log_probs, out_lengths = model_raw(mel, lengths)
            hyps = greedy_decode(log_probs.float(), out_lengths, blank_id)
            cum  = 0
            for i, ll in enumerate(label_lengths.tolist()):
                ref_toks = labels_flat[cum: cum + ll].tolist()
                ref_text = tokenizer.decode(ref_toks)
                hyp_text = tokenizer.decode(hyps[i])
                total_wer += word_error_rate(hyp_text, ref_text)
                n         += 1
                if len(examples) < 3:
                    examples.append((ref_text, hyp_text))
                cum += ll

    model_raw.train()
    return total_wer / max(n, 1), examples


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(output_dir, epoch, model, optimizer, scheduler):
    ckpt_dir = os.path.join(output_dir, f"checkpoint_ep{epoch:03d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, os.path.join(ckpt_dir, "state.pt"))
    all_ckpts = sorted(
        [c for c in os.listdir(output_dir) if c.startswith("checkpoint_ep")],
        key=lambda x: int(x.split("_ep")[-1]),
    )
    for old in all_ckpts[:-NUM_KEEP_CHECKPOINTS]:
        shutil.rmtree(os.path.join(output_dir, old))
    return ckpt_dir


def load_checkpoint(ckpt_dir, model, optimizer, scheduler):
    state = torch.load(os.path.join(ckpt_dir, "state.pt"),
                       map_location="cpu", weights_only=True)
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state["epoch"]


# ─────────────────────────────────────────────────────────────────────────────
# SUBSAMPLE BY HOURS
# ─────────────────────────────────────────────────────────────────────────────

def subsample_by_hours(dataset, hours, seed):
    import soundfile as _sf
    rng     = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    target, total, selected = hours * 3600.0, 0.0, []
    for idx in indices:
        path, _ = dataset.librispeech_data[idx]
        try:
            total += _sf.info(path).duration
        except Exception:
            continue
        selected.append(idx)
        if total >= target:
            break
    return Subset(dataset, selected)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)

    if rank == 0:
        print(f"Run : {args.run_name}  world={world_size}  "
              f"max_frames={MAX_MEL_FRAMES}  grad_accum={GRAD_ACCUM}  "
              f"eff_batch={args.batch_size * world_size * GRAD_ACCUM}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer   = BPETokenizer(args.bpe_model)
    num_classes = tokenizer.vocab_size   # 128
    blank_id    = tokenizer.pad_token_id  # 0
    if rank == 0:
        print(f"BPE vocab={num_classes}  blank={blank_id}")

    # ── Datasets ──────────────────────────────────────────────────────────
    if rank == 0:
        print("Scanning audio lengths…")
    full_train = LibriSpeechDataset(
        path_to_data_root=args.data_root,
        include_splits=TRAIN_SPLITS,
        tokenizer=tokenizer,
        hop_length=MEL_HOP,
        scan_lengths=True,
    )

    if args.hours is not None and args.hours < 960:
        train_dataset = subsample_by_hours(full_train, args.hours, args.seed)
        lengths = [full_train.lengths[i] for i in train_dataset.indices]
    else:
        train_dataset = full_train
        lengths = full_train.lengths

    lengths = [min(l, MAX_MEL_FRAMES) for l in lengths]

    if rank == 0:
        print(f"Train : {len(train_dataset):,} utterances")

    dev_clean = LibriSpeechDataset(
        path_to_data_root=args.data_root,
        include_splits=[DEV_CLEAN_SPLIT],
        tokenizer=tokenizer,
        hop_length=MEL_HOP,
        scan_lengths=False,
    )
    dev_other = LibriSpeechDataset(
        path_to_data_root=args.data_root,
        include_splits=[DEV_OTHER_SPLIT],
        tokenizer=tokenizer,
        hop_length=MEL_HOP,
        scan_lengths=False,
    )

    # ── Bucketed DataLoader ────────────────────────────────────────────────
    bucket_sampler = DistributedBucketBatchSampler(
        lengths               = lengths,
        batch_size            = args.batch_size,
        num_replicas          = world_size,
        rank                  = rank,
        bucket_size_multiplier= 100,
        shuffle               = True,
        seed                  = args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler      = bucket_sampler,
        num_workers        = args.workers,
        collate_fn         = collate_fn,
        pin_memory         = True,
        persistent_workers = True,
        prefetch_factor    = 4,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = get_config(VARIANT)
    model     = build_model(model_cfg, num_classes).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if rank == 0:
        print(f"SqueezeFormer-{VARIANT.upper()} | {n_params/1e6:.1f}M params | {world_size} GPU(s)")

    spec_augment = ConformerSpecAugment().to(device)

    # ── Optimiser + LR schedule ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = build_noam_scheduler(optimizer, WARMUP_EPOCHS, PEAK_EPOCHS)
    ctc_loss  = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    batches_per_epoch   = len(bucket_sampler)
    opt_steps_per_epoch = math.ceil(batches_per_epoch / GRAD_ACCUM)
    if rank == 0:
        print(f"batches/epoch={batches_per_epoch}  "
              f"opt_steps/epoch={opt_steps_per_epoch}  "
              f"warmup={WARMUP_EPOCHS}ep  peak={PEAK_EPOCHS}ep")

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume is not None:
        barrier(world_size)
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler) + 1
        if rank == 0:
            print(f"Resumed → epoch {start_epoch}")

    # ── Training loop ─────────────────────────────────────────────────────
    loss_window = collections.deque(maxlen=50)

    for epoch in range(start_epoch, args.epochs + 1):
        bucket_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        opt_steps  = 0

        if rank == 0:
            bar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", unit="batch",
                       dynamic_ncols=True)
        else:
            bar = train_loader

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(bar):
            data, lengths_b, labels_flat, label_lengths = batch

            mel           = data.to(device, non_blocking=True).float()
            lengths_b     = lengths_b.to(device, non_blocking=True)
            labels_flat   = labels_flat.to(device, non_blocking=True)
            label_lengths = label_lengths.to(device, non_blocking=True)

            is_last_accum = (
                (batch_idx + 1) % GRAD_ACCUM == 0 or
                batch_idx == batches_per_epoch - 1
            )

            # Skip allreduce on intermediate accum steps
            sync_ctx = (contextlib.nullcontext()
                        if (not hasattr(model, "no_sync") or is_last_accum)
                        else model.no_sync())

            with sync_ctx:
                with torch.no_grad():
                    mel = spec_augment(mel.transpose(1, 2)).transpose(1, 2)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    log_probs, out_lengths = model(mel, lengths_b)
                    log_probs_t = log_probs.transpose(0, 1).float()
                    loss = ctc_loss(log_probs_t, labels_flat,
                                    out_lengths, label_lengths)

                (loss / GRAD_ACCUM).backward()

            if is_last_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1

                if rank == 0:
                    lv = loss.item()
                    loss_window.append(lv)
                    epoch_loss += lv
                    bar.set_postfix(
                        loss   = f"{lv:.3f}",
                        smooth = f"{sum(loss_window)/len(loss_window):.3f}",
                        lr     = f"{scheduler.get_last_lr()[0]:.2e}",
                    )

        # ── Step LR once per epoch (Noam schedule) ────────────────────────
        scheduler.step()

        # ── Epoch summary (rank-0) ─────────────────────────────────────────
        if rank == 0:
            avg_loss  = epoch_loss / max(opt_steps, 1)
            model_raw = model.module if hasattr(model, "module") else model

            wer_clean, ex_clean = evaluate_dev(
                model_raw, dev_clean, tokenizer, blank_id, device)
            wer_other, _        = evaluate_dev(
                model_raw, dev_other, tokenizer, blank_id, device)

            print(f"\n[ep{epoch:03d}] loss={avg_loss:.4f}  "
                  f"dev-clean={wer_clean*100:.2f}%  dev-other={wer_other*100:.2f}%  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
            for ref_txt, hyp_txt in ex_clean:
                print(f"  REF: {ref_txt}")
                print(f"  HYP: {hyp_txt}")
            print()

            if epoch % SAVE_EVERY == 0:
                ckpt = save_checkpoint(
                    args.output_dir, epoch, model, optimizer, scheduler)
                print(f"  [ckpt] → {ckpt}")

        barrier(world_size)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
