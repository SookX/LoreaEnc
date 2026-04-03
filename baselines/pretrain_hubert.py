"""
pretrain_hubert.py
──────────────────
Pre-trains a HuBERT-base model FROM RANDOM INITIALISATION on the full
LibriSpeech 960 h (same unlabeled corpus Lorea uses).

HuBERT requires offline discrete targets (k-means cluster IDs over MFCC).
Steps:
  1. Fit MiniBatchKMeans on MFCC features extracted from a subset of audio.
  2. Assign cluster IDs to every training utterance.
  3. Pre-train with masked-prediction loss over those cluster IDs.

Launch with torchrun (see slurm/pretrain_hubert.sh):
    torchrun --nproc_per_node=8 baselines/pretrain_hubert.py
"""

import sys
import os
import argparse
import random
import logging
import pickle

import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset, DistributedSampler
from transformers import (
    HubertModel,
    HubertConfig,
    Wav2Vec2FeatureExtractor,
)
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.dataset import LibriSpeechDataset
from dataset.collate_waveform import collate_fn_waveform
from configs.pretrain_config import PretrainConfig
from trainers.pretrain_trainer import PretrainTrainer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HuBERT-base architecture (exact match to facebook/hubert-base-ls960)
# ─────────────────────────────────────────────────────────────────────────────

HUBERT_BASE_CONFIG = dict(
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
    feat_extract_norm="group",
    feat_extract_activation="gelu",
    conv_dim=(512, 512, 512, 512, 512, 512, 512),
    conv_stride=(5, 2, 2, 2, 2, 2, 2),
    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
    conv_bias=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# MFCC extraction — hop_length=320 matches CNN stride so label count aligns
# ─────────────────────────────────────────────────────────────────────────────

def extract_mfcc(waveform: np.ndarray, sr: int = 16_000, n_mfcc: int = 39) -> np.ndarray:
    """waveform: 1-D float32 → [T_frames, n_mfcc].

    hop_length=320 matches HuBERT's CNN feature extractor stride (5×2^6=320),
    so label frame count aligns with transformer output frame count.
    """
    t = torch.from_numpy(waveform).float().unsqueeze(0)
    mfcc_tf = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 320, "n_mels": 80},
    )
    mfcc = mfcc_tf(t)                    # [1, n_mfcc, T_frames]
    return mfcc.squeeze(0).T.numpy()     # [T_frames, n_mfcc]


# ─────────────────────────────────────────────────────────────────────────────
# K-means fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_or_load_kmeans(dataset, cache_dir: str, n_clusters: int, sr: int,
                       smoke_test: bool = False):
    """Fit MiniBatchKMeans on MFCC features or load from cache."""
    from sklearn.cluster import MiniBatchKMeans

    os.makedirs(cache_dir, exist_ok=True)
    km_path = os.path.join(cache_dir, f"kmeans_k{n_clusters}.pkl")

    if os.path.isfile(km_path):
        logger.info(f"Loading k-means from cache: {km_path}")
        with open(km_path, "rb") as f:
            return pickle.load(f)

    logger.info(f"Fitting MiniBatchKMeans (k={n_clusters}) …")
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=4096,
        random_state=42,
        n_init=5,
        max_iter=200,
    )

    n_fit = min(len(dataset), 10 if smoke_test else 5_000)
    indices = random.sample(range(len(dataset)), n_fit)
    for idx in indices:
        sample = dataset[idx]
        wav = sample["raw_audio"].squeeze(0).numpy()
        km.partial_fit(extract_mfcc(wav, sr=sr))

    if not smoke_test:
        with open(km_path, "wb") as f:
            pickle.dump(km, f)
        logger.info(f"K-means saved → {km_path}")

    return km


def assign_labels(waveform: np.ndarray, km, sr: int) -> np.ndarray:
    return km.predict(extract_mfcc(waveform, sr=sr)).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Span masking helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_span_mask(batch_size: int, seq_len: int, mask_prob: float = 0.065,
                    mask_span: int = 10, device=None) -> torch.BoolTensor:
    """Return BoolTensor [B, T] where True = masked position."""
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    num_starts = max(1, int(seq_len * mask_prob))
    for b in range(batch_size):
        starts = torch.randperm(max(1, seq_len - mask_span), device=device)[:num_starts]
        for s in starts:
            mask[b, s : s + mask_span] = True
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# HuBERT pre-training wrapper (randomly initialised)
# ─────────────────────────────────────────────────────────────────────────────

class HubertForPreTraining(nn.Module):
    """
    Masked-prediction wrapper around HubertModel.
    Generates span masks, passes them to the encoder (so masked frames see
    no input signal), and computes cross-entropy only at masked positions.
    """

    def __init__(self, config):
        super().__init__()
        self.hubert = HubertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_class_ids)
        self.config = config

    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        mask_time_indices = None
        if labels is not None:
            feat_len = self.hubert._get_feat_extract_output_lengths(
                attention_mask.sum(-1) if attention_mask is not None
                else torch.full(
                    (input_values.shape[0],),
                    input_values.shape[-1],
                    device=input_values.device,
                )
            ).min().item()
            mask_time_indices = _make_span_mask(
                input_values.shape[0], int(feat_len),
                mask_prob=0.065, mask_span=10,
                device=input_values.device,
            )

        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            output_attentions=kwargs.get("output_attentions"),
            output_hidden_states=kwargs.get("output_hidden_states"),
            return_dict=True,
        )
        logits = self.classifier(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            min_t = min(logits.shape[1], labels.shape[1])
            logits_t = logits[:, :min_t, :].contiguous()
            labels_t = labels[:, :min_t].contiguous()
            # Compute loss only at masked positions
            mask_t = mask_time_indices[:, :min_t]
            masked_labels = labels_t.masked_fill(~mask_t, -100)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits_t.view(-1, self.config.num_class_ids),
                masked_labels.view(-1),
            )

        return type("HubertOutput", (), {"loss": loss, "logits": logits})()

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.hubert.save_pretrained(save_directory)
        torch.save(
            self.classifier.state_dict(),
            os.path.join(save_directory, "classifier.bin"),
        )
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_config(cls, config):
        return cls(config)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset with k-means labels
# ─────────────────────────────────────────────────────────────────────────────

class HubertLabeledDataset(Dataset):
    def __init__(self, base_dataset, km, sr: int = 16_000):
        self.base = base_dataset
        self.km = km
        self.sr = sr

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        wav = sample["raw_audio"].squeeze(0).numpy()
        sample["hubert_labels"] = torch.from_numpy(assign_labels(wav, self.km, self.sr))
        return sample


def hubert_collate_fn(batch, feature_extractor, sampling_rate):
    base_out = collate_fn_waveform(
        batch, feature_extractor=feature_extractor, sampling_rate=sampling_rate,
    )
    label_seqs = [s["hubert_labels"] for s in batch]
    max_len = max(l.shape[0] for l in label_seqs)
    padded = torch.full((len(label_seqs), max_len), fill_value=-100, dtype=torch.long)
    for i, lab in enumerate(label_seqs):
        padded[i, : lab.shape[0]] = lab
    base_out["labels"] = padded
    return base_out


class HubertCollator:
    def __init__(self, feature_extractor, sampling_rate: int = 16_000):
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate

    def __call__(self, batch):
        return hubert_collate_fn(batch, self.feature_extractor, self.sampling_rate)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> PretrainConfig:
    cfg = PretrainConfig(output_dir="./outputs/hubert_pretrained")
    p = argparse.ArgumentParser(description="HuBERT pre-training from scratch")
    p.add_argument("--data_root",              type=str,   default=cfg.data_root)
    p.add_argument("--output_dir",             type=str,   default=cfg.output_dir)
    p.add_argument("--hubert_kmeans_cache",    type=str,   default=cfg.hubert_kmeans_cache)
    p.add_argument("--hubert_kmeans_clusters", type=int,   default=cfg.hubert_kmeans_clusters)
    p.add_argument("--max_steps",              type=int,   default=cfg.max_steps)
    p.add_argument("--batch_size",             type=int,   default=cfg.batch_size)
    p.add_argument("--grad_accum_steps",       type=int,   default=cfg.grad_accum_steps)
    p.add_argument("--lr",                     type=float, default=cfg.lr)
    p.add_argument("--warmup_steps",           type=int,   default=cfg.warmup_steps)
    p.add_argument("--precision",              type=str,   default=cfg.precision,
                   choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed",                   type=int,   default=cfg.seed)
    p.add_argument("--save_steps",             type=int,   default=cfg.save_steps)
    p.add_argument("--eval_steps",             type=int,   default=cfg.eval_steps)
    p.add_argument("--log_steps",              type=int,   default=cfg.log_steps)
    p.add_argument("--num_workers",            type=int,   default=cfg.num_workers)
    p.add_argument("--smoke_test",             action="store_true")
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

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if is_main:
        logger.info("=== HuBERT pre-training from random initialisation ===")
        logger.info(f"World size: {world_size} | Precision: {cfg.precision}")

    # ── Feature extractor ─────────────────────────────────────────────────
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/hubert-base-ls960"
    )

    # ── Base dataset for k-means (only rank-0 fits, others load cache) ────
    base_train = LibriSpeechDataset(
        path_to_data_root=cfg.data_root,
        include_splits=cfg.train_splits,
        sampling_rate=cfg.sampling_rate,
        mode="waveform",
    )
    base_eval = LibriSpeechDataset(
        path_to_data_root=cfg.data_root,
        include_splits=["dev-clean"],
        sampling_rate=cfg.sampling_rate,
        mode="waveform",
    )

    if cfg.smoke_test:
        base_train = Subset(base_train, [0, 10, 20, 30])
        base_eval  = Subset(base_eval,  [0, 1])

    # K-means is fit on rank-0 and saved; other ranks load from cache.
    if is_main:
        km = fit_or_load_kmeans(
            base_train, cache_dir=cfg.hubert_kmeans_cache,
            n_clusters=cfg.hubert_kmeans_clusters,
            sr=cfg.sampling_rate, smoke_test=cfg.smoke_test,
        )
    if world_size > 1:
        dist.barrier()   # wait for rank-0 to save the cache
    if not is_main:
        km = fit_or_load_kmeans(
            base_train, cache_dir=cfg.hubert_kmeans_cache,
            n_clusters=cfg.hubert_kmeans_clusters,
            sr=cfg.sampling_rate, smoke_test=cfg.smoke_test,
        )

    # ── Model (random init) ────────────────────────────────────────────────
    config = HubertConfig(**HUBERT_BASE_CONFIG)
    config.num_class_ids = cfg.hubert_kmeans_clusters
    model = HubertForPreTraining.from_config(config)

    if is_main:
        n = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {n:,} (randomly initialised)")

    # ── Labeled datasets ───────────────────────────────────────────────────
    train_ds = HubertLabeledDataset(base_train, km, sr=cfg.sampling_rate)
    eval_ds  = HubertLabeledDataset(base_eval,  km, sr=cfg.sampling_rate)

    collate = HubertCollator(feature_extractor, cfg.sampling_rate)

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
