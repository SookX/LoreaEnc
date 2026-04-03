"""
finetune.py
───────────
CTC fine-tuning of a locally pre-trained wav2vec2 or HuBERT checkpoint.
Uses the exact same labeled manifest as Lorea for fair comparison.

Usage:
    # Fine-tune wav2vec2 on the 10 h labeled subset:
    torchrun --nproc_per_node=8 baselines/finetune.py \\
        --model wav2vec2 \\
        --checkpoint ./outputs/wav2vec2_pretrained/checkpoint-400000 \\
        --train_manifest ./dataset/splits/10h.json \\
        --output_dir ./outputs/wav2vec2_finetuned_10h

    # Fine-tune HuBERT on 1 h:
    torchrun --nproc_per_node=8 baselines/finetune.py \\
        --model hubert \\
        --checkpoint ./outputs/hubert_pretrained/checkpoint-400000 \\
        --train_manifest ./dataset/splits/1h.json \\
        --output_dir ./outputs/hubert_finetuned_1h

    # Smoke test:
    torchrun --nproc_per_node=1 baselines/finetune.py \\
        --model wav2vec2 \\
        --checkpoint ./outputs/wav2vec2_pretrained/checkpoint-400000 \\
        --train_manifest ./dataset/splits/10h.json \\
        --smoke_test
"""

import sys
import os
import argparse
import random
import logging

import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, DistributedSampler
from transformers import (
    Wav2Vec2ForCTC,
    HubertForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.dataset import LibriSpeechDataset
from dataset.collate_waveform import collate_fn_waveform
from configs.finetune_config import FinetuneConfig
from trainers.finetune_trainer import FinetuneTrainer

logger = logging.getLogger(__name__)

MODEL_CLS = {
    "wav2vec2": Wav2Vec2ForCTC,
    "hubert":   HubertForCTC,
}


# ─────────────────────────────────────────────────────────────────────────────
# Processor
# ─────────────────────────────────────────────────────────────────────────────

def build_processor(checkpoint: str) -> Wav2Vec2Processor:
    """
    Build a Wav2Vec2Processor from the pre-trained checkpoint.
    For models trained from scratch the checkpoint directory will contain
    the feature extractor config saved by WaveformCollator / save_pretrained.
    Falls back to the HF wav2vec2-base tokenizer (English character vocab).
    """
    try:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(checkpoint)
    except Exception:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            "facebook/wav2vec2-base",
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
    except Exception:
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16_000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# ─────────────────────────────────────────────────────────────────────────────
# CTC collator
# ─────────────────────────────────────────────────────────────────────────────

class CtcCollator:
    def __init__(self, processor, sampling_rate: int = 16_000):
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __call__(self, batch):
        base = collate_fn_waveform(
            batch,
            feature_extractor=self.processor.feature_extractor,
            sampling_rate=self.sampling_rate,
        )
        encoded = self.processor.tokenizer(
            base["raw_transcripts"], padding=True, return_tensors="pt"
        )
        labels = encoded.input_ids.masked_fill(encoded.attention_mask.eq(0), -100)
        base["labels"] = labels
        return base


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> FinetuneConfig:
    cfg = FinetuneConfig()
    p = argparse.ArgumentParser(description="CTC fine-tuning (wav2vec2 / HuBERT)")
    p.add_argument("--model",            type=str, required=True,
                   choices=["wav2vec2", "hubert"])
    p.add_argument("--checkpoint",       type=str, required=True,
                   help="Path to locally pre-trained checkpoint directory")
    p.add_argument("--train_manifest",   type=str, required=True,
                   help="Path to labeled split JSON (dataset/splits/*.json)")
    p.add_argument("--data_root",        type=str, default=cfg.data_root)
    p.add_argument("--eval_split",       type=str, default=cfg.eval_split)
    p.add_argument("--output_dir",       type=str, default=None)
    p.add_argument("--max_steps",        type=int, default=cfg.max_steps)
    p.add_argument("--batch_size",       type=int, default=cfg.batch_size)
    p.add_argument("--grad_accum_steps", type=int, default=cfg.grad_accum_steps)
    p.add_argument("--lr",               type=float, default=cfg.lr)
    p.add_argument("--warmup_ratio",     type=float, default=cfg.warmup_ratio)
    p.add_argument("--precision",        type=str, default=cfg.precision,
                   choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed",             type=int, default=cfg.seed)
    p.add_argument("--save_steps",       type=int, default=cfg.save_steps)
    p.add_argument("--eval_steps",       type=int, default=cfg.eval_steps)
    p.add_argument("--log_steps",        type=int, default=cfg.log_steps)
    p.add_argument("--num_workers",      type=int, default=cfg.num_workers)
    p.add_argument("--smoke_test",       action="store_true")
    args = p.parse_args()

    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    cfg.model_name_or_path = args.checkpoint
    cfg.train_manifest = args.train_manifest
    cfg._model_type = args.model

    if args.output_dir is None:
        label_tag = os.path.splitext(os.path.basename(args.train_manifest))[0]
        cfg.output_dir = f"./outputs/{args.model}_finetuned_{label_tag}"

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── DDP init ──────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    is_main = (local_rank == 0)

    cfg = parse_args()

    random.seed(cfg.seed + local_rank)
    np.random.seed(cfg.seed + local_rank)
    torch.manual_seed(cfg.seed + local_rank)
    torch.cuda.manual_seed_all(cfg.seed + local_rank)

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if is_main:
        logger.info(f"=== CTC fine-tuning: {cfg._model_type} ===")
        logger.info(f"Checkpoint: {cfg.model_name_or_path}")
        logger.info(f"Train manifest: {cfg.train_manifest}")
        logger.info(f"World size: {world_size} | Precision: {cfg.precision}")

    # ── Processor & model ─────────────────────────────────────────────────
    processor = build_processor(cfg.model_name_or_path)

    cls = MODEL_CLS[cfg._model_type]
    model = cls.from_pretrained(
        cfg.model_name_or_path,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
        ctc_zero_infinity=cfg.ctc_zero_infinity,
    )
    # Freeze CNN feature encoder — standard for CTC fine-tuning
    model.freeze_feature_encoder()

    if is_main:
        total  = sum(p.numel() for p in model.parameters())
        active = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Params: {total:,} total | {active:,} trainable")

    # ── Datasets ───────────────────────────────────────────────────────────
    # Train from the exact manifest that Lorea uses (ensures fair comparison)
    train_ds = LibriSpeechDataset(
        path_to_data_root=cfg.data_root,
        manifest_path=cfg.train_manifest,
        sampling_rate=cfg.sampling_rate,
        train_split=True,
        apply_spec_augment=False,
        apply_audio_augment=False,
        mode="waveform",
    )
    eval_ds = LibriSpeechDataset(
        path_to_data_root=cfg.data_root,
        include_splits=[cfg.eval_split],
        sampling_rate=cfg.sampling_rate,
        train_split=False,
        apply_spec_augment=False,
        apply_audio_augment=False,
        mode="waveform",
    )

    if cfg.smoke_test:
        train_ds = Subset(train_ds, list(range(min(4, len(train_ds)))))
        eval_ds  = Subset(eval_ds,  list(range(min(4, len(eval_ds)))))

    if is_main:
        logger.info(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    collate = CtcCollator(processor, cfg.sampling_rate)

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=local_rank,
            shuffle=True, drop_last=True,
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        shuffle=shuffle, num_workers=cfg.num_workers,
        collate_fn=collate, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True,
    ) if is_main else None

    # ── Trainer ────────────────────────────────────────────────────────────
    warmup_steps = int(cfg.warmup_ratio * cfg.max_steps)
    trainer = FinetuneTrainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        processor=processor,
        output_dir=cfg.output_dir,
        max_steps=cfg.max_steps,
        grad_accum_steps=cfg.grad_accum_steps,
        lr=cfg.lr,
        warmup_steps=warmup_steps,
        precision=cfg.precision,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        log_steps=cfg.log_steps,
        local_rank=local_rank,
        world_size=world_size,
        train_sampler=train_sampler,
        smoke_test=cfg.smoke_test,
    )

    trainer.train()

    # ── Save processor alongside model (rank-0 only) ──────────────────────
    if is_main and not cfg.smoke_test:
        processor.save_pretrained(cfg.output_dir)
        logger.info(f"Processor saved → {cfg.output_dir}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
