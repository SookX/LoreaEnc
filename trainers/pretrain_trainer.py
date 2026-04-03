"""
PretrainTrainer: self-supervised pre-training loop.
Works for both Wav2Vec2ForPreTraining and the custom HubertForPreTraining.
Both return .loss directly from forward(), so the logic is model-agnostic.
"""
import logging
import inspect
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class PretrainTrainer(BaseTrainer):
    """
    Pre-training trainer.

    wav2vec2: loss = contrastive_loss + diversity_loss  (computed internally)
    HuBERT:   loss = cross-entropy over k-means cluster targets (masked positions only)

    The trainer is model-agnostic: calls model(**batch) and reads .loss.
    The caller is responsible for putting the correct keys in the batch dict.
    """

    def __init__(self, model, train_loader, eval_loader=None, **kwargs):
        super().__init__(
            model, train_loader, eval_loader,
            find_unused_parameters=True,   # quantizer / classifier may be sparse
            **kwargs,
        )
        # Cache the set of keys accepted by the underlying model's forward()
        raw = self.model.module if hasattr(self.model, "module") else self.model
        self._forward_keys = set(inspect.signature(raw.forward).parameters.keys())

    def _filter_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v for k, v in batch.items()
            if k in self._forward_keys and isinstance(v, torch.Tensor)
        }

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        outputs = self.model(**self._filter_batch(batch))
        metrics = {"loss": outputs.loss}
        if getattr(outputs, "contrastive_loss", None) is not None:
            metrics["contrastive_loss"] = outputs.contrastive_loss
        if getattr(outputs, "diversity_loss", None) is not None:
            metrics["diversity_loss"] = outputs.diversity_loss
        return metrics

    def eval_loop(self) -> Dict[str, float]:
        """Short eval pass on rank-0. Returns average loss over up to 50 batches."""
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.eval_loader:
                if n >= 50:
                    break
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with torch.amp.autocast(
                    device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
                ):
                    outputs = self.model(**self._filter_batch(batch))
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                    n += 1
        return {"eval_loss": total_loss / max(n, 1)}
