"""
evaluate.py
───────────
Standalone WER evaluation on a saved fine-tuned CTC checkpoint.

Usage:
    python baselines/evaluate.py --model wav2vec2 \
        --checkpoint ./outputs/wav2vec2_finetuned

    python baselines/evaluate.py --model hubert \
        --checkpoint ./outputs/hubert_finetuned \
        --eval_split dev-clean

    python baselines/evaluate.py --model wav2vec2 \
        --checkpoint facebook/wav2vec2-base-960h \
        --eval_split test-clean
"""

import sys
import os
import argparse
import json
import re
import logging
from typing import List

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from transformers import (
    Wav2Vec2ForCTC,
    HubertForCTC,
    Wav2Vec2Processor,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.dataset import LibriSpeechDataset
from dataset.collate_waveform import WaveformCollator

logger = logging.getLogger(__name__)

MODEL_CLS = {
    "wav2vec2": Wav2Vec2ForCTC,
    "hubert":   HubertForCTC,
}


# ─────────────────────────────────────────────────────────────────────────────
# WER
# ─────────────────────────────────────────────────────────────────────────────

def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    try:
        import evaluate as hf_evaluate
        wer_metric = hf_evaluate.load("wer")
        return wer_metric.compute(predictions=hypotheses, references=references)
    except Exception:
        pass
    total_w, total_e = 0, 0
    for ref, hyp in zip(references, hypotheses):
        rw, hw = ref.split(), hyp.split()
        total_w += len(rw)
        r, h = len(rw), len(hw)
        dp = list(range(h + 1))
        for i in range(1, r + 1):
            nd = [i] + [0] * h
            for j in range(1, h + 1):
                nd[j] = dp[j-1] if rw[i-1] == hw[j-1] else 1 + min(dp[j], nd[j-1], dp[j-1])
            dp = nd
        total_e += dp[h]
    return total_e / max(total_w, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="WER evaluation of a fine-tuned CTC model")
    p.add_argument("--model",      type=str, required=True, choices=["wav2vec2", "hubert"])
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to fine-tuned directory or HF model id")
    p.add_argument("--data_root",  type=str, default="./dataset")
    p.add_argument("--eval_split", type=str, default="dev-clean")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--fp16",       type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--output_json",type=str, default=None,
                   help="If given, write results dict to this JSON file")
    p.add_argument("--smoke_test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Evaluating {args.model} @ {args.checkpoint} on {args.eval_split}")
    logger.info(f"Device: {device}")

    # ── Load processor + model ────────────────────────────────────────────
    processor = Wav2Vec2Processor.from_pretrained(args.checkpoint)
    cls = MODEL_CLS[args.model]
    model = cls.from_pretrained(args.checkpoint).to(device).eval()

    # ── Dataset ───────────────────────────────────────────────────────────
    ds = LibriSpeechDataset(
        path_to_data_root=args.data_root,
        include_splits=args.eval_split,
        sampling_rate=16_000,
        train_split=False,
        apply_spec_augment=False,
        apply_audio_augment=False,
        mode="waveform",
    )
    if args.smoke_test:
        ds = Subset(ds, list(range(min(8, len(ds)))))

    collate = WaveformCollator(
        feature_extractor=processor.feature_extractor,
        sampling_rate=16_000,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate, pin_memory=True,
    )

    # ── Inference ─────────────────────────────────────────────────────────
    all_refs, all_hyps = [], []
    with torch.no_grad():
        for batch in loader:
            input_values  = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast(enabled=args.fp16):
                logits = model(input_values=input_values,
                               attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1)
            decoded = processor.batch_decode(pred_ids)

            refs = [re.sub(r"[^\w\s]", "", r.upper().strip())
                    for r in batch["raw_transcripts"]]
            hyps = [re.sub(r"[^\w\s]", "", h.upper().strip())
                    for h in decoded]
            all_refs.extend(refs)
            all_hyps.extend(hyps)

    wer = compute_wer(all_refs, all_hyps)
    logger.info(f"WER on {args.eval_split}: {wer*100:.2f}%")

    # Print 5 examples
    logger.info("\n─── Sample predictions ───")
    for ref, hyp in zip(all_refs[:5], all_hyps[:5]):
        logger.info(f"  REF: {ref}")
        logger.info(f"  HYP: {hyp}")
        logger.info("")

    results = {
        "model":      args.model,
        "checkpoint": args.checkpoint,
        "eval_split": args.eval_split,
        "wer":        wer,
        "wer_pct":    round(wer * 100, 2),
        "n_samples":  len(all_refs),
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results written → {args.output_json}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
