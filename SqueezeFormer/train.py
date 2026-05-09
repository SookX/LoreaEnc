"""
SqueezeFormer XS — CTC training on LibriSpeech.

Single GPU:
    python SqueezeFormer/train.py

Multi-GPU (2 GPUs):
    torchrun --nproc_per_node=2 SqueezeFormer/train.py

With label-fraction override (used by SLURM):
    torchrun --nproc_per_node=2 SqueezeFormer/train.py \
        --hours 1 --seed 42 --output-dir outputs/baseline/1h --run-name baseline_1h

Resume:
    torchrun --nproc_per_node=2 SqueezeFormer/train.py \
        --resume outputs/baseline/1h/checkpoint_ep010
"""

import os
import sys
import math
import random
import shutil
import argparse
import collections
import socket
import time

import torch
import torch.nn as nn
import torchaudio
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
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
                   help="Suppress a parameter update if the pre-clipping grad norm exceeds this value. "
                        "Use 0 to disable.")
    p.add_argument("--workers",    type=int,   default=NUM_WORKERS)
    p.add_argument("--eval-split", type=str,   default=DEV_SPLIT,
                   help="LibriSpeech split for validation WER.")
    p.add_argument("--eval-every", type=int,   default=EVAL_EVERY,
                   help="Run full validation every N epochs.")
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="Debug/probe mode: stop each epoch after N training batches.")
    p.add_argument("--log-every", type=int, default=100,
                   help="Print rank-0 training heartbeat every N batches. Use 0 to disable.")
    p.add_argument("--debug-ranks", action="store_true",
                   help="Print rank-stamped diagnostics around DDP setup, batches, and barriers.")
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


# ──────────────────────────────────────────────────────────────────────────────
# DISTRIBUTED / CHECKPOINTS
# ──────────────────────────────────────────────────────────────────────────────

def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)
        return 0, 0, 1, device

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return rank, local_rank, world_size, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def print0(rank: int, *args, **kwargs):
    if is_main_process(rank):
        print(*args, **kwargs, flush=True)


def debug_print(enabled: bool, rank: int, message: str):
    if enabled:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        host = socket.gethostname()
        print(f"[{now}] [host {host}] [rank {rank}] {message}", flush=True)


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor.item()


def strip_state_prefixes(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def unwrap_model(model):
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_wer):
    os.makedirs(path, exist_ok=True)
    module = unwrap_model(model)
    torch.save(
        {
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_wer": best_wer,
        },
        os.path.join(path, "checkpoint.pt"),
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """Load native DDP checkpoints, or old Accelerate checkpoints for resume."""
    native_path = os.path.join(path, "checkpoint.pt") if os.path.isdir(path) else path
    if os.path.isfile(native_path):
        state = torch.load(native_path, map_location=device)
        model.load_state_dict(strip_state_prefixes(state["model"]), strict=True)
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        return state.get("epoch"), state.get("best_wer")

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    safetensors_path = os.path.join(path, "model.safetensors")
    torch_model_path = os.path.join(path, "pytorch_model.bin")
    if os.path.isfile(safetensors_path):
        from safetensors.torch import load_file
        model_state = load_file(safetensors_path, device=str(device))
    elif os.path.isfile(torch_model_path):
        model_state = torch.load(torch_model_path, map_location=device)
    else:
        raise FileNotFoundError(f"No model checkpoint found in {path}")

    model.load_state_dict(strip_state_prefixes(model_state), strict=True)

    optimizer_path = os.path.join(path, "optimizer.bin")
    if optimizer is not None and os.path.isfile(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

    scheduler_path = os.path.join(path, "scheduler.bin")
    if scheduler is not None and os.path.isfile(scheduler_path):
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))

    return None, None


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
def evaluate_dev_other(model, dev_loader, tokenizer, blank_id, device, ctc_loss=None, rank=0):
    """Full greedy CTC evaluation on the configured dev split."""
    model.eval()

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
        disable=not is_main_process(rank),
    ):
        mel = mel.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            log_probs, out_lengths = model(mel, lengths)
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
            if is_main_process(rank) and example is None:
                example = (hyp, ref)

    if dist.is_available() and dist.is_initialized():
        totals = torch.tensor(
            [total_word_errors, total_ref_words, total_loss, total_batches],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_word_errors = totals[0].item()
        total_ref_words = totals[1].item()
        total_loss = totals[2].item()
        total_batches = totals[3].item()

    wer = total_word_errors / max(total_ref_words, 1)
    avg_loss = total_loss / max(total_batches, 1) if ctc_loss is not None else None
    model.train()
    return {"wer": wer, "loss": avg_loss, "example": example}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main_ddp():
    args = parse_args()

    run_name = args.run_name or f"{EXPERIMENT_NAME}_{args.variant}_{args.eval_split}"
    output_dir = args.output_dir or os.path.join(WORKING_DIR, run_name)
    seed = args.seed
    hours = args.hours
    grad_accum_steps = max(1, args.grad_accum_steps)

    debug_print(args.debug_ranks, -1, "entering setup_distributed")
    rank, local_rank, world_size, device = setup_distributed()
    debug_print(
        args.debug_ranks,
        rank,
        f"distributed ready local_rank={local_rank} world_size={world_size} device={device} "
        f"cuda_devices={torch.cuda.device_count()}",
    )
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    debug_print(args.debug_ranks, rank, "building tokenizer")
    tokenizer = build_tokenizer(args.tokenizer_path)
    blank_id = tokenizer.pad_token_id
    num_classes = tokenizer.vocab_size
    print0(rank, f"Vocab size: {num_classes} | CTC blank: {blank_id}")
    if args.tokenizer_path is None:
        print0(
            rank,
            "Tokenizer: facebook/wav2vec2-base fallback. For paper-faithful training, "
            "pass --tokenizer-path to a 128-token SentencePiece model.",
        )

    debug_print(args.debug_ranks, rank, "building train dataset")
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
        print0(rank, f"Subsampling to {hours}h (seed={seed}) ...")
        train_dataset = subsample_by_hours(full_train, hours, seed)
        print0(rank, f"  -> {len(train_dataset)} utterances selected")
    else:
        train_dataset = full_train
        print0(rank, f"Using full dataset: {len(train_dataset)} utterances")

    debug_print(args.debug_ranks, rank, "building dev dataset")
    dev_dataset = LibriSpeechDataset(
        path_to_data_root=args.data_root,
        include_splits=[args.eval_split],
        tokenizer=tokenizer,
        train_split=False,
        apply_spec_augment=False,
        mode="mel",
    )

    train_sampler = None
    dev_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        dev_sampler = DistributedSampler(
            dev_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    debug_print(args.debug_ranks, rank, "building train dataloader")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    debug_print(args.debug_ranks, rank, "building dev dataloader")
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size or args.batch_size,
        shuffle=False,
        sampler=dev_sampler,
        num_workers=args.workers,
        collate_fn=eval_collate_fn,
        pin_memory=True,
    )
    print0(rank, f"Validation: {args.eval_split} | {len(dev_dataset)} utterances")

    debug_print(args.debug_ranks, rank, "building model")
    model_config = get_config(args.variant)
    model = build_model(model_config, num_classes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print0(
        rank,
        f"SqueezeFormer-{args.variant.upper()} | {n_params/1e6:.1f}M params | "
        f"{world_size} GPU(s) | run: {run_name}",
    )

    peak_lr = args.lr if args.lr is not None else paper_peak_lr(args.variant)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
    )

    num_epochs = args.epochs
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    scheduler = build_extended_noam_scheduler(
        optimizer=optimizer,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=NUM_WARMUP_EPOCHS,
        peak_epochs=NUM_PEAK_EPOCHS,
        decay_rate=NOAM_DECAY_RATE,
    )
    print0(
        rank,
        f"LR: extended Noam | peak={peak_lr:g} | warmup={NUM_WARMUP_EPOCHS} ep | "
        f"hold={NUM_PEAK_EPOCHS} ep | decay={NOAM_DECAY_RATE:g} | train={num_epochs} ep",
    )
    print0(
        rank,
        f"Batching: per_gpu_micro={args.batch_size} | grad_accum={grad_accum_steps} | "
        f"effective_global={args.batch_size * grad_accum_steps * world_size} | "
        f"optimizer_steps/epoch={steps_per_epoch}",
    )

    ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    start_epoch = 1
    best_wer = float("inf")
    if args.resume is not None:
        loaded_epoch, loaded_best_wer = load_checkpoint(args.resume, model, optimizer, scheduler, device=device)
        if args.start_epoch is not None:
            start_epoch = args.start_epoch
        elif loaded_epoch is not None:
            start_epoch = loaded_epoch + 1
        else:
            try:
                start_epoch = int(args.resume.rstrip("/").split("_ep")[-1]) + 1
            except ValueError as exc:
                raise ValueError(
                    "--start-epoch is required when --resume does not end with _epNNN "
                    f"(got {args.resume!r})"
                ) from exc
        if loaded_best_wer is not None:
            best_wer = loaded_best_wer
        print0(rank, f"Resumed from {args.resume} -> starting epoch {start_epoch}")

    debug_print(args.debug_ranks, rank, "moving model to device")
    model.to(device)
    move_optimizer_state(optimizer, device)
    if sys.platform != "win32" and not args.no_compile:
        model = torch.compile(model)
        print0(rank, "torch.compile: enabled")
    else:
        print0(rank, "torch.compile: skipped")

    if world_size > 1:
        debug_print(args.debug_ranks, rank, "wrapping model in DDP")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    debug_print(args.debug_ranks, rank, "training loop ready")

    epoch_bar = tqdm(
        range(start_epoch, num_epochs + 1),
        desc=f"{run_name} | {n_params/1e6:.1f}M | {world_size} GPU(s)",
        unit="ep",
        dynamic_ncols=True,
        position=0,
        disable=not is_main_process(rank),
    )

    loss_window = collections.deque(maxlen=50)
    last_grad_norm = float("nan")

    try:
        for epoch in epoch_bar:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            epoch_ter = 0.0
            n_batches = 0

            batch_bar = tqdm(
                train_loader,
                desc=f"  Epoch {epoch:03d}",
                unit="batch",
                dynamic_ncols=True,
                position=1,
                leave=False,
                disable=not is_main_process(rank),
            )

            for step, batch in enumerate(batch_bar, start=1):
                if args.max_train_batches is not None and step > args.max_train_batches:
                    break

                if step <= 3:
                    debug_print(args.debug_ranks, rank, f"epoch {epoch} got batch {step}")
                mel, lengths, labels, label_lengths = batch
                mel = mel.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                label_lengths = label_lengths.to(device, non_blocking=True)

                sync_step = (step % grad_accum_steps == 0) or (step == len(train_loader))
                sync_context = model.no_sync if isinstance(model, DDP) and not sync_step else torch.enable_grad

                with sync_context():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        log_probs, output_lengths = model(mel, lengths)
                        loss = ctc_loss(log_probs.permute(1, 0, 2), labels, output_lengths, label_lengths)
                        if not torch.isfinite(loss):
                            print0(rank, f"[warn] non-finite loss at epoch {epoch}, batch {step}; using zero-loss update")
                            loss = log_probs.sum() * 0.0
                        scaled_loss = loss / grad_accum_steps
                    scaled_loss.backward()

                if sync_step:
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                    last_grad_norm = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    grad_is_safe = (
                        math.isfinite(last_grad_norm)
                        and (args.max_safe_grad_norm <= 0.0 or last_grad_norm <= args.max_safe_grad_norm)
                    )
                    if not grad_is_safe:
                        print0(
                            rank,
                            f"[warn] unsafe grad norm {last_grad_norm:.2f} at epoch {epoch}, "
                            f"batch {step}; suppressing parameter update",
                        )
                        for param in model.parameters():
                            param.grad = None
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if args.log_every > 0 and is_main_process(rank) and (
                        step <= 4 or step % args.log_every == 0 or step == len(train_loader)
                    ):
                        print(
                            f"[train] epoch={epoch} batch={step}/{len(train_loader)} "
                            f"loss={loss.detach().float().item():.4f} grad={last_grad_norm:.2f} "
                            f"lr={scheduler.get_last_lr()[0]:.3e}",
                            flush=True,
                        )

                loss_val = loss.detach().float().item()
                loss_window.append(loss_val)
                epoch_loss += loss_val
                n_batches += 1

                if is_main_process(rank):
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

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_ter = epoch_ter / max(n_batches, 1)

            should_eval = (args.eval_every > 0) and (epoch % args.eval_every == 0 or epoch == num_epochs)
            if should_eval and dev_sampler is not None:
                dev_sampler.set_epoch(epoch)
            if should_eval:
                debug_print(args.debug_ranks, rank, f"epoch {epoch} starting evaluation")
                eval_metrics = evaluate_dev_other(
                    model=model,
                    dev_loader=dev_loader,
                    tokenizer=tokenizer,
                    blank_id=blank_id,
                    device=device,
                    ctc_loss=ctc_loss,
                    rank=rank,
                )
            if is_main_process(rank) and should_eval:
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

                if improved:
                    ckpt_dir = os.path.join(output_dir, "checkpoint_best")
                    save_checkpoint(ckpt_dir, model, optimizer, scheduler, epoch, best_wer)
                    tqdm.write(f"  [ckpt] best {args.eval_split} WER saved -> {ckpt_dir}")
            elif is_main_process(rank):
                epoch_bar.set_postfix_str(
                    f"loss {avg_loss:.4f}  avg-TER {avg_ter:.2%}"
                    f"  lr {scheduler.get_last_lr()[0]:.1e}",
                    refresh=True,
                )
            barrier()

            if epoch % SAVE_EVERY == 0:
                if is_main_process(rank):
                    ckpt_dir = os.path.join(output_dir, f"checkpoint_ep{epoch:03d}")
                    save_checkpoint(ckpt_dir, model, optimizer, scheduler, epoch, best_wer)
                    tqdm.write(f"  [ckpt] saved -> {ckpt_dir}")

                    all_ckpts = sorted(
                        [c for c in os.listdir(output_dir) if c.startswith("checkpoint_ep")],
                        key=lambda x: int(x.split("_ep")[-1]),
                    )
                    for old in all_ckpts[:-NUM_KEEP_CHECKPOINTS]:
                        shutil.rmtree(os.path.join(output_dir, old))
                barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    record(main_ddp)()
