"""
pretrain_wav2vec2.py
────────────────────
Pre-trains a wav2vec2-base model FROM RANDOM INITIALISATION on the full
LibriSpeech 960 h (same unlabeled corpus Lorea uses).

Launch with torchrun (see slurm/pretrain_wav2vec2.sh):
    torchrun --nproc_per_node=8 baselines/pretrain_wav2vec2.py

Key flags:
    --data_root     path to dataset dir       (default: ./dataset)
    --output_dir    where to save             (default: ./outputs/wav2vec2_pretrained)
    --max_steps     gradient steps            (default: 400 000)
    --batch_size    per-GPU batch             (default: 32)
    --lr            learning rate             (default: 5e-4)
    --precision     bf16 | fp16 | fp32        (default: bf16)
    --smoke_test    4 samples, 2 steps, no save
"""

import sys
import os
import argparse
import random
import logging

import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForPreTraining,
    Wav2Vec2FeatureExtractor,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.dataset import LibriSpeechDataset
from dataset.collate_waveform import WaveformCollator
from configs.pretrain_config import PretrainConfig
from trainers.pretrain_trainer import PretrainTrainer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# wav2vec2-base architecture (exact match to facebook/wav2vec2-base)
# ─────────────────────────────────────────────────────────────────────────────

WAV2VEC2_BASE_CONFIG = dict(
    vocab_size=32,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout=0.1,
    activation_dropout=0.0,
    attention_dropout=0.1,
    feat_proj_dropout=0.0,
    layerdrop=0.05,
    initializer_range=0.02,
    num_conv_pos_embeddings=128,
    num_conv_pos_embedding_groups=16,
    do_stable_layer_norm=False,
    apply_spec_augment=True,
    mask_time_prob=0.065,
    mask_time_length=10,
    mask_time_min_masks=2,
    mask_feature_prob=0.0,
    mask_feature_length=10,
    num_codevectors_per_group=320,
    num_codevector_groups=2,
    contrastive_logits_temperature=0.1,
    num_negatives=100,
    codevector_dim=256,
    proj_codevector_dim=256,
    diversity_loss_weight=0.1,
    feat_extract_norm="group",
    feat_extract_activation="gelu",
    conv_dim=(512, 512, 512, 512, 512, 512, 512),
    conv_stride=(5, 2, 2, 2, 2, 2, 2),
    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
    conv_bias=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> PretrainConfig:
    cfg = PretrainConfig(output_dir="./outputs/wav2vec2_pretrained")
    p = argparse.ArgumentParser(description="wav2vec2 pre-training from scratch")
    p.add_argument("--data_root",        type=str,   default=cfg.data_root)
    p.add_argument("--output_dir",       type=str,   default=cfg.output_dir)
    p.add_argument("--max_steps",        type=int,   default=cfg.max_steps)
    p.add_argument("--batch_size",       type=int,   default=cfg.batch_size)
    p.add_argument("--grad_accum_steps", type=int,   default=cfg.grad_accum_steps)
    p.add_argument("--lr",               type=float, default=cfg.lr)
    p.add_argument("--warmup_steps",     type=int,   default=cfg.warmup_steps)
    p.add_argument("--precision",        type=str,   default=cfg.precision,
                   choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed",             type=int,   default=cfg.seed)
    p.add_argument("--save_steps",       type=int,   default=cfg.save_steps)
    p.add_argument("--eval_steps",       type=int,   default=cfg.eval_steps)
    p.add_argument("--log_steps",        type=int,   default=cfg.log_steps)
    p.add_argument("--num_workers",      type=int,   default=cfg.num_workers)
    p.add_argument("--smoke_test",       action="store_true")
    args = p.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
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
        logger.info("=== wav2vec2 pre-training from random initialisation ===")
        logger.info(f"World size: {world_size} | Precision: {cfg.precision}")

    # ── Model (random init) ────────────────────────────────────────────────
    config = Wav2Vec2Config(**WAV2VEC2_BASE_CONFIG)
    config.mask_time_prob   = cfg.mask_time_prob
    config.mask_time_length = cfg.mask_time_length
    model = Wav2Vec2ForPreTraining(config)

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        logger.info(f"Model parameters: {n_params:,} (randomly initialised)")

    # ── Feature extractor (processor only, no pretrained weights) ─────────
    # We need the feature extractor to normalise waveforms; its parameters
    # are fixed (mean/variance normalisation) — no learned weights.
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base"
    )

    # ── Datasets ───────────────────────────────────────────────────────────
    train_ds = LibriSpeechDataset(
        path_to_data_root=cfg.data_root,
        include_splits=cfg.train_splits,   # all 960 h
        sampling_rate=cfg.sampling_rate,
        train_split=False,
        apply_spec_augment=False,
        apply_audio_augment=False,
        mode="waveform",
    )
    eval_ds = LibriSpeechDataset(
        path_to_data_root=cfg.data_root,
        include_splits=["dev-clean"],
        sampling_rate=cfg.sampling_rate,
        train_split=False,
        apply_spec_augment=False,
        apply_audio_augment=False,
        mode="waveform",
    )

    if cfg.smoke_test:
        train_ds = Subset(train_ds, [0, 10, 20, 30])
        eval_ds  = Subset(eval_ds,  [0, 1])

    # ── DataLoaders ────────────────────────────────────────────────────────
    collate = WaveformCollator(feature_extractor, cfg.sampling_rate)

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
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    # Eval runs on rank-0 only
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
    ) if is_main else None

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = PretrainTrainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        output_dir=cfg.output_dir,
        max_steps=cfg.max_steps,
        grad_accum_steps=cfg.grad_accum_steps,
        lr=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        precision=cfg.precision,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        log_steps=cfg.log_steps,
        local_rank=local_rank,
        world_size=world_size,
        train_sampler=train_sampler,
        smoke_test=cfg.smoke_test,
    )

    out = trainer.train()
    if is_main:
        logger.info(f"Pre-training complete. Checkpoint: {out}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
