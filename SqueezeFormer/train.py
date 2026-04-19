"""
SqueezeFormer XS — CTC training on LibriSpeech.

Single GPU:
    python SqueezeFormer/train.py

Multi-GPU (2 GPUs):
    accelerate launch --num_processes 2 SqueezeFormer/train.py

With label-fraction override (used by SLURM):
    accelerate launch --num_processes 2 SqueezeFormer/train.py \
        --hours 1 --seed 42 --output-dir outputs/baseline/1h --run-name baseline_1h

Resume:
    accelerate launch --num_processes 2 SqueezeFormer/train.py \
        --resume outputs/baseline/1h/checkpoint_ep010
"""

import os
import sys
import math
import random
import shutil
import argparse
import collections

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from transformers import Wav2Vec2CTCTokenizer, get_scheduler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.dataset import LibriSpeechDataset
from SqueezeFormer import Squeezeformer, get_config

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS — overridden by CLI flags when launched from SLURM
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME  = "SqueezeFormer_XS_train100h"
WORKING_DIR      = "./work_dir"
DATA_ROOT        = "./dataset"

TRAIN_SPLITS          = ["train-clean-100", "train-clean-360", "train-other-500"]
DEV_SPLIT             = "dev-clean"
NUM_WORKERS           = 4           # set to 0 on Windows if RAM is tight

VARIANT               = "xs"

LEARNING_RATE         = 5e-4
WEIGHT_DECAY          = 1e-6
ADAM_BETA1            = 0.9
ADAM_BETA2            = 0.98
ADAM_EPSILON          = 1e-8

PER_GPU_BATCH_SIZE    = 32
GRADIENT_ACCUMULATION = 1
NUM_EPOCHS            = 150
NUM_WARMUP_EPOCHS     = 10
LR_SCHEDULER_TYPE     = "cosine"

SAVE_EVERY            = 10
NUM_KEEP_CHECKPOINTS  = 5
LOG_WANDB             = False
SEED                  = 42


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hours",      type=float, default=None,
                   help="Subsample training set to N hours (reproducible via --seed).")
    p.add_argument("--seed",       type=int,   default=SEED,
                   help="Random seed for data subsampling and training.")
    p.add_argument("--output-dir", type=str,   default=None,
                   help="Override default experiment output directory.")
    p.add_argument("--run-name",   type=str,   default=None,
                   help="Override experiment name (used in W&B and checkpoint paths).")
    p.add_argument("--resume",     type=str,   default=None,
                   help="Path to checkpoint directory to resume from.")
    p.add_argument("--data-root",  type=str,   default=DATA_ROOT,
                   help="Root directory of LibriSpeech.")
    p.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int,   default=PER_GPU_BATCH_SIZE)
    p.add_argument("--workers",    type=int,   default=NUM_WORKERS)
    p.add_argument("--wandb",      action="store_true", default=LOG_WANDB)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def subsample_by_hours(dataset: LibriSpeechDataset, hours: float, seed: int) -> Subset:
    """
    Shuffle the dataset with `seed`, then greedily take utterances until
    `hours` of audio is accumulated. Uses torchaudio.info() — reads file
    headers only, no audio loaded into memory.
    """
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    target_seconds = hours * 3600.0
    accumulated    = 0.0
    selected       = []

    for idx in indices:
        path, _ = dataset.librispeech_data[idx]
        try:
            info = torchaudio.info(path)
            dur  = info.num_frames / info.sample_rate
        except Exception:
            continue
        selected.append(idx)
        accumulated += dur
        if accumulated >= target_seconds:
            break

    return Subset(dataset, selected)


def collate_fn(batch):
    mel_list   = [item["input_values"] for item in batch]
    label_list = [item["labels"]       for item in batch]
    lengths    = torch.tensor([m.size(0) for m in mel_list], dtype=torch.long)

    mel_padded    = nn.utils.rnn.pad_sequence(mel_list, batch_first=True)
    label_lengths = torch.tensor([len(l) for l in label_list], dtype=torch.long)
    labels_flat   = torch.cat([
        l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in label_list
    ])
    return mel_padded, lengths, labels_flat, label_lengths


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config, num_classes: int) -> Squeezeformer:
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
# DECODING / METRICS
# ─────────────────────────────────────────────────────────────────────────────

def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int):
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


def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def token_error_rate(hyps, refs):
    total_err, total_ref = 0, 0
    for h, r in zip(hyps, refs):
        total_err += _edit_distance(h, r)
        total_ref += max(len(r), 1)
    return total_err / max(total_ref, 1)


def word_error_rate(hyp_text: str, ref_text: str) -> float:
    hyp_words = hyp_text.lower().split()
    ref_words = ref_text.lower().split()
    return _edit_distance(hyp_words, ref_words) / max(len(ref_words), 1)


def evaluate_sample(model, dev_dataset, tokenizer, blank_id, accelerator):
    """Decode one random dev-clean sample on the main process only."""
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()

    idx    = random.randrange(len(dev_dataset))
    sample = dev_dataset[idx]
    mel    = sample["input_values"].unsqueeze(0).to(accelerator.device)
    length = torch.tensor([mel.size(1)], dtype=torch.long, device=accelerator.device)
    ref    = sample["raw_transcript"]

    with torch.no_grad():
        log_probs, out_lengths = unwrapped(mel, length)
        hyp_tokens = greedy_decode(log_probs, out_lengths, blank_id)[0]

    hyp = tokenizer.decode(hyp_tokens)
    wer = word_error_rate(hyp, ref)
    unwrapped.train()
    return hyp, ref, wer


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve experiment name and output dir from CLI or defaults
    run_name       = args.run_name   or EXPERIMENT_NAME
    output_dir     = args.output_dir or os.path.join(WORKING_DIR, run_name)
    seed           = args.seed
    hours          = args.hours      # None → use full dataset
    os.makedirs(output_dir, exist_ok=True)

    # ── Accelerator ───────────────────────────────────────────────────────
    accelerator = Accelerator(
        project_dir=output_dir,
        log_with="wandb" if args.wandb else None,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        mixed_precision="bf16",   # H200: 989 TFLOPs BF16 vs 67 TFLOPs FP32
    )
    if args.wandb:
        accelerator.init_trackers(run_name)

    torch.manual_seed(seed)
    random.seed(seed)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer   = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
    blank_id    = tokenizer.pad_token_id
    num_classes = tokenizer.vocab_size
    accelerator.print(f"Vocab size: {num_classes} | CTC blank: {blank_id}")

    # ── Datasets ──────────────────────────────────────────────────────────
    full_train = LibriSpeechDataset(
        path_to_data_root=args.data_root,
        include_splits=TRAIN_SPLITS,
        tokenizer=tokenizer,
        train_split=True,
        apply_spec_augment=True,
        mode="mel",
    )

    if hours is not None:
        accelerator.print(f"Subsampling to {hours}h (seed={seed}) …")
        train_dataset = subsample_by_hours(full_train, hours, seed)
        accelerator.print(f"  → {len(train_dataset)} utterances selected")
    else:
        train_dataset = full_train
        accelerator.print(f"Using full dataset: {len(train_dataset)} utterances")

    dev_dataset = LibriSpeechDataset(
        path_to_data_root=args.data_root,
        include_splits=[DEV_SPLIT],
        tokenizer=tokenizer,
        train_split=False,
        apply_spec_augment=False,
        mode="mel",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // GRADIENT_ACCUMULATION,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.workers > 0,  # keep workers alive between epochs
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model_config = get_config(VARIANT)
    model        = build_model(model_config, num_classes)
    n_params     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(
        f"SqueezeFormer-{VARIANT.upper()} | {n_params/1e6:.1f}M params | "
        f"{accelerator.num_processes} GPU(s) | run: {run_name}"
    )

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
    )

    num_epochs       = args.epochs
    steps_per_epoch  = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION)
    total_steps      = num_epochs * steps_per_epoch
    warmup_steps     = NUM_WARMUP_EPOCHS * steps_per_epoch

    scheduler = get_scheduler(
        name=LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    # ── SyncBatchNorm: required for correct BN stats across GPUs ──────────
    if accelerator.num_processes > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # ── torch.compile: fused kernels, significant H200 speedup ───────────
    # Requires gcc/triton — Linux/HPC only, skipped on Windows
    if sys.platform != "win32":
        model = torch.compile(model)
        accelerator.print("torch.compile: enabled")
    else:
        accelerator.print("torch.compile: skipped (Windows)")

    # ── Hand everything to Accelerate ─────────────────────────────────────
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    accelerator.register_for_checkpointing(scheduler)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume is not None:
        with accelerator.main_process_first():
            accelerator.load_state(args.resume)
        start_epoch = int(args.resume.rstrip("/").split("_ep")[-1]) + 1
        accelerator.print(f"Resumed from {args.resume} → starting epoch {start_epoch}")

    # ── Training loop ─────────────────────────────────────────────────────
    epoch_bar = tqdm(
        range(start_epoch, num_epochs + 1),
        desc=f"{run_name} | {n_params/1e6:.1f}M | {accelerator.num_processes} GPU(s)",
        unit="ep",
        dynamic_ncols=True,
        position=0,
        disable=not accelerator.is_local_main_process,
    )

    loss_window = collections.deque(maxlen=50)

    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        epoch_ter  = 0.0
        n_batches  = 0

        batch_bar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch:03d}",
            unit="batch",
            dynamic_ncols=True,
            position=1,
            leave=False,
            disable=not accelerator.is_local_main_process,
        )

        for batch in batch_bar:
            mel, lengths, labels, label_lengths = batch

            with accelerator.accumulate(model):
                log_probs, output_lengths = model(mel, lengths)
                loss = ctc_loss(log_probs.permute(1, 0, 2), labels, output_lengths, label_lengths)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loss_val = accelerator.gather(loss.detach().unsqueeze(0)).mean().item()
            loss_window.append(loss_val)
            epoch_loss += loss_val
            n_batches  += 1

            if accelerator.is_main_process:
                with torch.no_grad():
                    hyps = greedy_decode(
                        accelerator.unwrap_model(model)(mel, lengths)[0].detach(),
                        output_lengths, blank_id,
                    )
                offset, refs = 0, []
                for ll in label_lengths.tolist():
                    refs.append(labels[offset:offset + ll].tolist())
                    offset += ll
                ter = token_error_rate(hyps, refs)
                epoch_ter += ter

                smoothed = sum(loss_window) / len(loss_window)
                batch_bar.set_postfix_str(
                    f"loss {loss_val:.3f}  smooth {smoothed:.3f}  TER {ter:.2%}"
                    f"  lr {scheduler.get_last_lr()[0]:.1e}",
                    refresh=False,
                )

        # ── End-of-epoch ──────────────────────────────────────────────────
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_ter  = epoch_ter  / max(n_batches, 1)

        if accelerator.is_main_process:
            hyp, ref, wer = evaluate_sample(model, dev_dataset, tokenizer, blank_id, accelerator)
            tqdm.write(
                f"\n  [epoch {epoch:03d}]"
                f"\n    REF : {ref}"
                f"\n    HYP : {hyp}"
                f"\n    WER : {wer:.2%}\n"
            )
            epoch_bar.set_postfix_str(
                f"loss {avg_loss:.4f}  avg-TER {avg_ter:.2%}  WER {wer:.2%}"
                f"  lr {scheduler.get_last_lr()[0]:.1e}",
                refresh=True,
            )
            if args.wandb:
                accelerator.log({"loss": avg_loss, "TER": avg_ter, "WER": wer}, step=epoch)

        # ── Checkpoint ────────────────────────────────────────────────────
        if epoch % SAVE_EVERY == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt_dir = os.path.join(output_dir, f"checkpoint_ep{epoch:03d}")
                accelerator.save_state(output_dir=ckpt_dir)
                tqdm.write(f"  [ckpt] saved → {ckpt_dir}")

                all_ckpts = sorted(
                    [c for c in os.listdir(output_dir) if c.startswith("checkpoint_ep")],
                    key=lambda x: int(x.split("_ep")[-1]),
                )
                for old in all_ckpts[:-NUM_KEEP_CHECKPOINTS]:
                    shutil.rmtree(os.path.join(output_dir, old))

            accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
