"""
Pre-training configuration dataclass.
Defaults tuned for 8×H200 (141 GB HBM3e each) on a single node.
Covers both wav2vec2 and HuBERT pre-training from random initialisation.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class PretrainConfig:
    # ── Model ──────────────────────────────────────────────────────────────
    model_name_or_path: str = "wav2vec2-base"
    """Architecture tag used to select the config; no pretrained weights loaded."""

    # ── Data ───────────────────────────────────────────────────────────────
    data_root: str = "./dataset"
    train_splits: List[str] = field(
        default_factory=lambda: ["train-clean-100", "train-clean-360", "train-other-500"]
    )
    """Full 960 h — identical to what Lorea will see during pre-training."""
    subset_fraction: float = 1.0
    """1.0 = full split. Reduce only for quick experiments."""
    num_workers: int = 4
    """Per-GPU DataLoader workers. 4 is safe on most HPC nodes."""
    sampling_rate: int = 16_000

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir: str = "./outputs/wav2vec2_pretrained"

    # ── Training ───────────────────────────────────────────────────────────
    max_steps: int = 400_000
    """Original wav2vec2 paper uses 400 k steps on 32 GPUs.
    With 8×H200 and batch_size=32 the effective batch (256 samples) is similar."""
    batch_size: int = 32
    """Per-GPU batch.  Effective = batch_size × world_size (no grad accum needed)."""
    grad_accum_steps: int = 1
    lr: float = 5e-4
    warmup_steps: int = 32_000
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    """'bf16' (recommended on H200), 'fp16', or 'fp32'."""
    seed: int = 42

    # ── Masking (wav2vec2) ─────────────────────────────────────────────────
    mask_time_prob: float = 0.065
    mask_time_length: int = 10

    # ── Checkpointing / Logging ────────────────────────────────────────────
    save_steps: int = 10_000
    eval_steps: int = 10_000
    log_steps: int = 200

    # ── Smoke test ─────────────────────────────────────────────────────────
    smoke_test: bool = False

    # ── HuBERT-specific ────────────────────────────────────────────────────
    hubert_kmeans_clusters: int = 100
    hubert_kmeans_cache: str = "./outputs/hubert_kmeans"
