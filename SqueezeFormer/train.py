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
from transformers import Wav2Vec2CTCTokenizer
from tqdm import tqdm

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    def record(fn):
        return fn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.dataset import LibriSpeechDataset
from SqueezeFormer import Squeezeformer, get_config

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS — overridden by CLI flags when launched from SLURM
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME  = "SqueezeFormer_ASR_150ep"
WORKING_DIR      = "./work_dir"
DATA_ROOT        = "./dataset"

TRAIN_SPLITS          = ["train-clean-100", "train-clean-360", "train-other-500"]
DEV_SPLIT             = "dev-other"
NUM_WORKERS           = 4           # set to 0 on Windows if RAM is tight

VARIANT               = "sm"

LEARNING_RATE         = None
WEIGHT_DECAY          = 5e-4
ADAM_BETA1            = 0.9
ADAM_BETA2            = 0.98
ADAM_EPSILON          = 1e-9

PER_GPU_BATCH_SIZE    = 32
GRADIENT_ACCUMULATION = 1
NUM_EPOCHS            = 150
NUM_WARMUP_EPOCHS     = 20
NUM_PEAK_EPOCHS       = 160
NOAM_DECAY_RATE       = 1.0

SAVE_EVERY            = 10
EVAL_EVERY            = 1
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
    p.add_argument("--start-epoch", type=int,  default=None,
                   help="Epoch number to start from after --resume. "
                        "Required for checkpoint names without _epNNN, such as checkpoint_best.")
    p.add_argument("--data-root",  type=str,   default=DATA_ROOT,
                   help="Root directory of LibriSpeech.")
    p.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int,   default=PER_GPU_BATCH_SIZE)
    p.add_argument("--grad-accum-steps", type=int, default=GRADIENT_ACCUMULATION,
                   help="Optimizer update every N microbatches. --batch-size is per-GPU microbatch size.")
    p.add_argument("--eval-batch-size", type=int, default=None,
                   help="Evaluation batch size. Defaults to --batch-size.")
    p.add_argument("--max-grad-norm", type=float, default=5.0,
                   help="Global gradient clipping norm applied on optimizer steps.")
    p.add_argument("--max-safe-grad-norm", type=float, default=0.0,
                   help="Skip an optimizer update if the pre-clipping grad norm exceeds this value. "
                        "Use 0 to disable.")
    p.add_argument("--workers",    type=int,   default=NUM_WORKERS)
    p.add_argument("--eval-split", type=str,   default=DEV_SPLIT,
                   help="LibriSpeech split for validation WER.")
    p.add_argument("--eval-every", type=int,   default=EVAL_EVERY,
                   help="Run full validation every N epochs.")
    p.add_argument("--variant",    type=str,   default=VARIANT,
                   choices=["xs", "s", "sm", "m", "ml", "l"])
    p.add_argument("--lr",         type=float, default=LEARNING_RATE,
                   help="Peak learning rate. Defaults to the paper value for the selected variant.")
    p.add_argument("--tokenizer-path", type=str, default=None,
                   help="SentencePiece model path. The paper uses a 128-token SentencePiece vocab.")
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile even on Linux.")
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


def eval_collate_fn(batch):
    mel_padded, lengths, labels_flat, label_lengths = collate_fn(batch)
    transcripts = [item["raw_transcript"] for item in batch]
    return mel_padded, lengths, labels_flat, label_lengths, transcripts


class SentencePieceCTCTokenizer:
    """Tiny CTC wrapper around a SentencePiece model."""

    def __init__(self, model_path: str):
        import sentencepiece as spm

        self.processor = spm.SentencePieceProcessor(model_file=model_path)
        self.piece_size = self.processor.get_piece_size()
        self.pad_token_id = 0
        self.vocab_size = self.piece_size

    def encode(self, text: str):
        return self.processor.encode(text.strip(), out_type=int)

    def decode(self, token_ids):
        pieces = [int(t) for t in token_ids if 0 <= int(t) < self.piece_size]
        return self.processor.decode(pieces)


def build_tokenizer(tokenizer_path: str = None):
    if tokenizer_path is None:
        return Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
    return SentencePieceCTCTokenizer(tokenizer_path)


def paper_peak_lr(variant: str) -> float:
    if variant in {"xs", "s", "sm"}:
        return 2e-3
    if variant in {"m", "ml"}:
        return 1.5e-3
    return 1e-3


def paper_specaugment_time_masks(variant: str) -> int:
    if variant == "m":
        return 7
    if variant in {"ml", "l"}:
        return 10
    return 5


def build_extended_noam_scheduler(
    optimizer,
    steps_per_epoch: int,
    warmup_epochs: int,
    peak_epochs: int,
    decay_rate: float,
):
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    peak_steps = max(0, peak_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        step = max(1, step)
        if step < warmup_steps:
            return step / warmup_steps
        if step < warmup_steps + peak_steps:
            return 1.0
        decay_step = step - peak_steps
        return (warmup_steps / max(decay_step, 1)) ** decay_rate

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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


def normalize_transcript(text: str) -> str:
    return " ".join(text.lower().strip().split())


@torch.no_grad()
def evaluate_dev_other(model, dev_loader, tokenizer, blank_id, accelerator, ctc_loss=None):
    """Full greedy CTC evaluation on the configured dev split."""
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()

    total_word_errors = 0
    total_ref_words = 0
    total_loss = 0.0
    total_batches = 0
    example = None

    for mel, lengths, labels, label_lengths, transcripts in tqdm(
        dev_loader,
        desc="  dev-other",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not accelerator.is_local_main_process,
    ):
        mel = mel.to(accelerator.device)
        lengths = lengths.to(accelerator.device)
        labels = labels.to(accelerator.device)
        label_lengths = label_lengths.to(accelerator.device)

        log_probs, out_lengths = unwrapped(mel, lengths)
        if ctc_loss is not None:
            loss = ctc_loss(log_probs.permute(1, 0, 2), labels, out_lengths, label_lengths)
            total_loss += loss.item()
            total_batches += 1

        hyp_tokens = greedy_decode(log_probs, out_lengths, blank_id)
        hyps = [normalize_transcript(tokenizer.decode(tokens)) for tokens in hyp_tokens]
        refs = [normalize_transcript(text) for text in transcripts]

        for hyp, ref in zip(hyps, refs):
            hyp_words = hyp.split()
            ref_words = ref.split()
            total_word_errors += _edit_distance(hyp_words, ref_words)
            total_ref_words += max(len(ref_words), 1)
            if example is None:
                example = (hyp, ref)

    wer = total_word_errors / max(total_ref_words, 1)
    avg_loss = total_loss / max(total_batches, 1) if ctc_loss is not None else None
    unwrapped.train()
    return {"wer": wer, "loss": avg_loss, "example": example}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve experiment name and output dir from CLI or defaults
    run_name       = args.run_name   or f"{EXPERIMENT_NAME}_{args.variant}_{args.eval_split}"
    output_dir     = args.output_dir or os.path.join(WORKING_DIR, run_name)
    seed           = args.seed
    hours          = args.hours      # None → use full dataset
    grad_accum_steps = max(1, args.grad_accum_steps)
    os.makedirs(output_dir, exist_ok=True)

    # ── Accelerator ───────────────────────────────────────────────────────
    accelerator = Accelerator(
        project_dir=output_dir,
        log_with="wandb" if args.wandb else None,
        gradient_accumulation_steps=grad_accum_steps,
        mixed_precision="bf16",   # H200: 989 TFLOPs BF16 vs 67 TFLOPs FP32
    )
    if args.wandb:
        accelerator.init_trackers(run_name)

    torch.manual_seed(seed)
    random.seed(seed)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer   = build_tokenizer(args.tokenizer_path)
    blank_id    = tokenizer.pad_token_id
    num_classes = tokenizer.vocab_size
    accelerator.print(f"Vocab size: {num_classes} | CTC blank: {blank_id}")
    if args.tokenizer_path is None:
        accelerator.print(
            "Tokenizer: facebook/wav2vec2-base fallback. For paper-faithful training, "
            "pass --tokenizer-path to a 128-token SentencePiece model."
        )

    # ── Datasets ──────────────────────────────────────────────────────────
    full_train = LibriSpeechDataset(
        path_to_data_root=args.data_root,
        include_splits=TRAIN_SPLITS,
        tokenizer=tokenizer,
        train_split=True,
        apply_spec_augment=True,
        mode="mel",
        spec_augment_num_time_masks=paper_specaugment_time_masks(args.variant),
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
        include_splits=[args.eval_split],
        tokenizer=tokenizer,
        train_split=False,
        apply_spec_augment=False,
        mode="mel",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,  # keep workers alive between epochs
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=eval_collate_fn,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    accelerator.print(f"Validation: {args.eval_split} | {len(dev_dataset)} utterances")

    # ── Model ─────────────────────────────────────────────────────────────
    model_config = get_config(args.variant)
    model        = build_model(model_config, num_classes)
    n_params     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(
        f"SqueezeFormer-{args.variant.upper()} | {n_params/1e6:.1f}M params | "
        f"{accelerator.num_processes} GPU(s) | run: {run_name}"
    )

    # ── Optimiser ─────────────────────────────────────────────────────────
    peak_lr = args.lr if args.lr is not None else paper_peak_lr(args.variant)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
    )

    num_epochs       = args.epochs
    steps_per_epoch  = math.ceil(len(train_loader) / grad_accum_steps)
    scheduler = build_extended_noam_scheduler(
        optimizer=optimizer,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=NUM_WARMUP_EPOCHS,
        peak_epochs=NUM_PEAK_EPOCHS,
        decay_rate=NOAM_DECAY_RATE,
    )
    accelerator.print(
        f"LR: extended Noam | peak={peak_lr:g} | warmup={NUM_WARMUP_EPOCHS} ep | "
        f"hold={NUM_PEAK_EPOCHS} ep | decay={NOAM_DECAY_RATE:g} | train={num_epochs} ep"
    )
    accelerator.print(
        f"Batching: per_gpu_micro={args.batch_size} | grad_accum={grad_accum_steps} | "
        f"effective_global={args.batch_size * grad_accum_steps * accelerator.num_processes} | "
        f"optimizer_steps/epoch={steps_per_epoch}"
    )

    ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    # ── SyncBatchNorm: required for correct BN stats across GPUs ──────────
    if accelerator.num_processes > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # ── torch.compile: fused kernels, significant H200 speedup ───────────
    # Requires gcc/triton — Linux/HPC only, skipped on Windows
    if sys.platform != "win32" and not args.no_compile:
        model = torch.compile(model)
        accelerator.print("torch.compile: enabled")
    else:
        accelerator.print("torch.compile: skipped")

    # ── Hand everything to Accelerate ─────────────────────────────────────
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    accelerator.register_for_checkpointing(scheduler)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume is not None:
        with accelerator.main_process_first():
            accelerator.load_state(args.resume)
        if args.start_epoch is not None:
            start_epoch = args.start_epoch
        else:
            try:
                start_epoch = int(args.resume.rstrip("/").split("_ep")[-1]) + 1
            except ValueError as exc:
                raise ValueError(
                    "--start-epoch is required when --resume does not end with _epNNN "
                    f"(got {args.resume!r})"
                ) from exc
        accelerator.print(f"Resumed from {args.resume} → starting epoch {start_epoch}")

    best_wer = float("inf")

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
    last_grad_norm = float("nan")

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

        for step, batch in enumerate(batch_bar, start=1):
            mel, lengths, labels, label_lengths = batch

            with accelerator.accumulate(model):
                log_probs, output_lengths = model(mel, lengths)
                loss = ctc_loss(log_probs.permute(1, 0, 2), labels, output_lengths, label_lengths)
                if not torch.isfinite(loss):
                    accelerator.print(f"[warn] non-finite loss at epoch {epoch}, batch {step}; skipping update")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                    last_grad_norm = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    grad_is_safe = (
                        math.isfinite(last_grad_norm)
                        and (args.max_safe_grad_norm <= 0.0 or last_grad_norm <= args.max_safe_grad_norm)
                    )
                    if grad_is_safe:
                        optimizer.step()
                        scheduler.step()
                    else:
                        accelerator.print(
                            f"[warn] unsafe grad norm {last_grad_norm:.2f} at epoch {epoch}, "
                            f"batch {step}; skipping update"
                        )
                    optimizer.zero_grad(set_to_none=True)

            loss_val = accelerator.gather(loss.detach().unsqueeze(0)).mean().item()
            loss_window.append(loss_val)
            epoch_loss += loss_val
            n_batches  += 1

            if accelerator.is_main_process:
                with torch.no_grad():
                    hyps = greedy_decode(log_probs.detach(), output_lengths, blank_id)
                offset, refs = 0, []
                for ll in label_lengths.tolist():
                    refs.append(labels[offset:offset + ll].tolist())
                    offset += ll
                ter = token_error_rate(hyps, refs)
                epoch_ter += ter

                smoothed = sum(loss_window) / len(loss_window)
                batch_bar.set_postfix_str(
                    f"loss {loss_val:.3f}  smooth {smoothed:.3f}  TER {ter:.2%}"
                    f"  grad {last_grad_norm:.2f}  lr {scheduler.get_last_lr()[0]:.1e}",
                    refresh=False,
                )

        # ── End-of-epoch ──────────────────────────────────────────────────
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_ter  = epoch_ter  / max(n_batches, 1)

        eval_metrics = None
        should_eval = (args.eval_every > 0) and (epoch % args.eval_every == 0 or epoch == num_epochs)
        if should_eval:
            accelerator.wait_for_everyone()
        if accelerator.is_main_process and should_eval:
            eval_metrics = evaluate_dev_other(
                model=model,
                dev_loader=dev_loader,
                tokenizer=tokenizer,
                blank_id=blank_id,
                accelerator=accelerator,
                ctc_loss=ctc_loss,
            )
            wer = eval_metrics["wer"]
            dev_loss = eval_metrics["loss"]
            hyp, ref = eval_metrics["example"] or ("", "")
            improved = wer < best_wer
            best_wer = min(best_wer, wer)
            tqdm.write(
                f"\n  [epoch {epoch:03d}] {args.eval_split}"
                f"\n    dev loss : {dev_loss:.4f}"
                f"\n    WER      : {wer:.2%}  best {best_wer:.2%}"
                f"\n    REF      : {ref}"
                f"\n    HYP      : {hyp}\n"
            )
            epoch_bar.set_postfix_str(
                f"loss {avg_loss:.4f}  avg-TER {avg_ter:.2%}  {args.eval_split} WER {wer:.2%}"
                f"  lr {scheduler.get_last_lr()[0]:.1e}",
                refresh=True,
            )
            if args.wandb:
                accelerator.log(
                    {
                        "train/loss": avg_loss,
                        "train/TER": avg_ter,
                        f"{args.eval_split}/loss": dev_loss,
                        f"{args.eval_split}/WER": wer,
                        f"{args.eval_split}/best_WER": best_wer,
                    },
                    step=epoch,
                )

            if improved:
                ckpt_dir = os.path.join(output_dir, "checkpoint_best")
                accelerator.save_state(output_dir=ckpt_dir)
                tqdm.write(f"  [ckpt] best {args.eval_split} WER saved → {ckpt_dir}")
        elif accelerator.is_main_process:
            epoch_bar.set_postfix_str(
                f"loss {avg_loss:.4f}  avg-TER {avg_ter:.2%}"
                f"  lr {scheduler.get_last_lr()[0]:.1e}",
                refresh=True,
            )
        if should_eval:
            accelerator.wait_for_everyone()

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
    record(main)()
