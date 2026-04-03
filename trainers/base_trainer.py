"""
BaseTrainer: shared training loop for all experiments.

DDP-aware: wrap the model in DistributedDataParallel before passing to this
class, and call dist.init_process_group() in the launch script.
Handles: optimizer, scheduler, AMP (bf16/fp16/fp32), grad clipping,
         checkpointing (rank-0 only), CSV logging (rank-0 only).
"""
import os
import csv
import time
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int
) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)


class BaseTrainer:
    """
    Minimal but complete DDP training loop.

    Sub-classes must implement:
        train_step(batch) -> dict with key "loss" and optional extra metrics
        eval_loop()       -> dict with metric values  (called on rank-0 only)

    DDP usage:
        Call dist.init_process_group() before constructing this class.
        Pass local_rank and world_size from the environment.
        The model will be wrapped in DDP automatically.
        Pass the DistributedSampler as train_sampler so set_epoch() is called
        at the start of every epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        output_dir: str,
        max_steps: int = 400_000,
        grad_accum_steps: int = 1,
        lr: float = 5e-4,
        warmup_steps: int = 32_000,
        weight_decay: float = 1e-2,
        max_grad_norm: float = 1.0,
        precision: str = "bf16",
        save_steps: int = 10_000,
        eval_steps: int = 10_000,
        log_steps: int = 200,
        local_rank: int = 0,
        world_size: int = 1,
        train_sampler: Optional[DistributedSampler] = None,
        find_unused_parameters: bool = True,
        smoke_test: bool = False,
    ):
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = (local_rank == 0)
        self.train_sampler = train_sampler
        self.smoke_test = smoke_test

        self.max_steps = 2 if smoke_test else max_steps
        self.grad_accum_steps = 1 if smoke_test else grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.log_steps = log_steps

        self.output_dir = output_dir
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # ── Device ────────────────────────────────────────────────────────
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # ── Precision ─────────────────────────────────────────────────────
        # bf16: recommended on H200; no GradScaler needed (no overflow risk)
        # fp16: requires GradScaler
        # fp32: no autocast
        assert precision in ("bf16", "fp16", "fp32"), f"Unknown precision: {precision}"
        self.precision = precision
        self.use_amp = precision in ("bf16", "fp16")
        self.amp_dtype = (
            torch.bfloat16 if precision == "bf16"
            else torch.float16 if precision == "fp16"
            else None
        )
        self.scaler = GradScaler(enabled=(precision == "fp16"))

        # ── Model → DDP ───────────────────────────────────────────────────
        self.model = model.to(self.device)
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=find_unused_parameters,
            )

        # ── Rank-0 only: dirs + file logging ─────────────────────────────
        if self.is_main:
            os.makedirs(output_dir, exist_ok=True)
            self._setup_logging()

        # ── Optimizer & scheduler ─────────────────────────────────────────
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.max_steps,
        )

        # ── CSV metrics log ───────────────────────────────────────────────
        self.log_path = os.path.join(output_dir, "metrics.csv")
        self._csv_fields = None

    # ── Logging ────────────────────────────────────────────────────────────

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.output_dir, "train.log")),
            ],
            force=True,
        )

    def _log_metrics(self, step: int, metrics: Dict[str, Any]):
        if not self.is_main:
            return
        if self._csv_fields is None:
            self._csv_fields = ["step"] + list(metrics.keys())
            with open(self.log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._csv_fields).writeheader()
        with open(self.log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writerow(
                {"step": step, **metrics}
            )

    # ── Checkpointing ──────────────────────────────────────────────────────

    def _unwrapped(self) -> nn.Module:
        """Return the raw module regardless of DDP wrapping."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def save_checkpoint(self, step: int):
        if not self.is_main or self.smoke_test:
            return
        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        self._unwrapped().save_pretrained(ckpt_dir)
        logger.info(f"Saved checkpoint → {ckpt_dir}")

    # ── Abstract interface ─────────────────────────────────────────────────

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError

    def eval_loop(self) -> Dict[str, float]:
        return {}

    # ── Main training loop ─────────────────────────────────────────────────

    def train(self):
        if self.is_main:
            logger.info(
                f"Training | device={self.device} | precision={self.precision} "
                f"| world_size={self.world_size} | max_steps={self.max_steps} "
                f"| grad_accum={self.grad_accum_steps}"
            )

        self.model.train()
        step = 0
        epoch = 0
        accum_metrics: Dict[str, float] = {}
        accum_count = 0
        t0 = time.time()

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        data_iter = iter(self.train_loader)
        self.optimizer.zero_grad()

        while step < self.max_steps:
            # ── Fetch batch (cycle over dataset) ─────────────────────────
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # ── Move to device ────────────────────────────────────────────
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # ── Forward + backward ────────────────────────────────────────
            with torch.amp.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                metrics = self.train_step(batch)

            loss = metrics["loss"]
            if loss is None:
                if self.is_main:
                    logger.warning("loss is None — skipping batch")
                continue

            self.scaler.scale(loss / self.grad_accum_steps).backward()

            for k, v in metrics.items():
                if v is None:
                    continue
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                accum_metrics[k] = accum_metrics.get(k, 0.0) + val
            accum_count += 1

            # ── Optimizer step ────────────────────────────────────────────
            if accum_count % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                step += 1

                avg = {k: v / self.grad_accum_steps for k, v in accum_metrics.items()}
                accum_metrics = {}

                if self.is_main and step % self.log_steps == 0:
                    lr_now = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    logger.info(
                        f"step {step}/{self.max_steps} | "
                        + " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
                        + f" | lr={lr_now:.2e} | {elapsed:.0f}s"
                    )
                    self._log_metrics(step, {**avg, "lr": lr_now})

                if step % self.eval_steps == 0 and self.eval_loader is not None:
                    # Eval runs on rank 0 only to avoid DistributedSampler complexity
                    if self.is_main:
                        self.model.eval()
                        eval_metrics = self.eval_loop()
                        self.model.train()
                        if eval_metrics:
                            logger.info(
                                f"[EVAL step {step}] "
                                + " | ".join(
                                    f"{k}={v:.4f}" for k, v in eval_metrics.items()
                                )
                            )
                            self._log_metrics(step, eval_metrics)
                    # Sync all ranks so non-main ranks wait during eval
                    if dist.is_initialized():
                        dist.barrier()

                if step % self.save_steps == 0:
                    self.save_checkpoint(step)

        if self.is_main:
            logger.info("Training complete.")
        if not self.smoke_test:
            self.save_checkpoint(step)
        return self.output_dir
