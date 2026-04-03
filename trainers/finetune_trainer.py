"""
FinetuneTrainer: CTC fine-tuning on top of pre-trained wav2vec2 / HuBERT.
Evaluates WER on the eval split (rank-0 only).
"""
import logging
import re
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


def _compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """Word error rate. Uses the `evaluate` library when available."""
    try:
        import evaluate as hf_evaluate
        return hf_evaluate.load("wer").compute(
            predictions=hypotheses, references=references
        )
    except Exception:
        pass
    # Manual Levenshtein WER fallback
    total_w = total_e = 0
    for ref, hyp in zip(references, hypotheses):
        rw, hw = ref.split(), hyp.split()
        total_w += len(rw)
        r, h = len(rw), len(hw)
        dp = list(range(h + 1))
        for i in range(1, r + 1):
            nd = [i] + [0] * h
            for j in range(1, h + 1):
                nd[j] = (
                    dp[j - 1]
                    if rw[i - 1] == hw[j - 1]
                    else 1 + min(dp[j], nd[j - 1], dp[j - 1])
                )
            dp = nd
        total_e += dp[h]
    return total_e / max(total_w, 1)


class FinetuneTrainer(BaseTrainer):
    """
    CTC fine-tuning trainer.

    Expects batches with:
        input_values    : FloatTensor[B, T]
        attention_mask  : LongTensor[B, T]
        labels          : LongTensor[B, L]   — token ids (padded with -100)

    The model must be a *ForCTC variant (wav2vec2 or hubert).
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        processor,
        **kwargs,
    ):
        super().__init__(
            model, train_loader, eval_loader,
            find_unused_parameters=False,
            **kwargs,
        )
        self.processor = processor

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        outputs = self.model(
            input_values=batch["input_values"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {"loss": outputs.loss}

    def eval_loop(self) -> Dict[str, float]:
        """Greedy-decode + WER on up to 100 eval batches (rank-0 only)."""
        all_refs: List[str] = []
        all_hyps: List[str] = []

        with torch.no_grad():
            for i, batch in enumerate(self.eval_loader):
                if i >= 100 and not self.smoke_test:
                    break
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with torch.amp.autocast(
                    device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
                ):
                    logits = self.model(
                        input_values=batch["input_values"],
                        attention_mask=batch["attention_mask"],
                    ).logits

                pred_ids = torch.argmax(logits, dim=-1)
                decoded = self.processor.batch_decode(pred_ids)

                refs = [
                    re.sub(r"[^\w\s]", "", r.upper().strip())
                    for r in batch["raw_transcripts"]
                ]
                hyps = [
                    re.sub(r"[^\w\s]", "", h.upper().strip())
                    for h in decoded
                ]
                all_refs.extend(refs)
                all_hyps.extend(hyps)

        return {"eval_wer": _compute_wer(all_refs, all_hyps)}
