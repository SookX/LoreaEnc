"""
Fine-tuning (CTC) configuration dataclass.
Shared by both wav2vec2 and HuBERT CTC fine-tuning.
Defaults tuned for 8×H200.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FinetuneConfig:
    # ── Model ──────────────────────────────────────────────────────────────
    model_name_or_path: str = ""
    """Path to locally pre-trained checkpoint (from pretrain_wav2vec2 / pretrain_hubert)."""

    # ── Data ───────────────────────────────────────────────────────────────
    data_root: str = "./dataset"
    train_manifest: str = ""
    """Path to a split JSON produced by dataset/splits.py.
    Must match exactly the labeled subset Lorea uses for fair comparison.
    e.g. dataset/splits/10h.json"""
    eval_split: str = "dev-clean"
    num_workers: int = 4
    sampling_rate: int = 16_000

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir: str = "./outputs/wav2vec2_finetuned"

    # ── Training ───────────────────────────────────────────────────────────
    max_steps: int = 20_000
    """~20 epochs over 10 h of labeled data with batch_size=32×8 GPUs."""
    batch_size: int = 32
    grad_accum_steps: int = 1
    lr: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    ctc_zero_infinity: bool = True
    precision: str = "bf16"
    seed: int = 42

    # ── Checkpointing / Logging ────────────────────────────────────────────
    save_steps: int = 1_000
    eval_steps: int = 1_000
    log_steps: int = 100

    # ── Smoke test ─────────────────────────────────────────────────────────
    smoke_test: bool = False
