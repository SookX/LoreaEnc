"""Standalone eval for MelHuBERT-Transformer (mh9m) fine-tune checkpoints.

This is the eval counterpart of finetune_mh9m.py. The standard
evaluate_ctc.py builds a CausalSpecUnitCTC model which uses `self.fc` for
the CTC head; our standalone fine-tune script uses `self.head` on a
minimal MH9MCTCModel. Loading the fine-tune checkpoint into the standard
model leaves the head randomly initialised, which produces nonsense
hypotheses and WER above 100%. This script uses the matching model class
so checkpoint parameters load strictly, and writes an eval_results.json
in the same format evaluate_ctc.py would write.

Usage (single checkpoint):
    torchrun --nproc_per_node=4 -m CausalSpecUnit.eval_mh9m \
        --checkpoint outputs/causal_specunit/mh9m_ft/librilight_10h/seed42/checkpoint_best/checkpoint.pt \
        --data-root dataset/datasets/librispeech/LibriSpeech \
        --cmvn-path outputs/causal_specunit/targets_960h_c8/cmvn.pt \
        --tokenizer-path dataset/bpe128.model \
        --splits test-clean test-other \
        --output outputs/causal_specunit/mh9m_ft/librilight_10h/seed42/eval_results.json
"""

import argparse
import json
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from SqueezeFormer.train import build_tokenizer
from CausalSpecUnit.data import CTCSpecDataset, collate_eval
from CausalSpecUnit.finetune_mh9m import (
    MH9MCTCModel,
    evaluate,
    setup_dist,
    cleanup_dist,
    is_main,
    log,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--cmvn-path", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--splits", nargs="+", default=["test-clean", "test-other"])
    p.add_argument("--variant", default="mh9m")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", default=None,
                   help="Where to write eval_results.json. Defaults to <ckpt_dir>/../eval_results.json.")
    args = p.parse_args()

    rank, local_rank, world_size = setup_dist()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    log(rank, f"loading checkpoint: {args.checkpoint}")
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = sd.get("model", sd)
    saved_vocab = sd.get("vocab_size")
    saved_blank = sd.get("blank_id")
    saved_variant = sd.get("variant", args.variant)
    log(rank, f"  saved vocab_size={saved_vocab} blank_id={saved_blank} variant={saved_variant}")

    tokenizer = build_tokenizer(args.tokenizer_path)
    vocab_size = tokenizer.vocab_size
    blank_id = vocab_size
    if saved_vocab is not None and saved_vocab != vocab_size:
        log(rank, f"  WARN: tokenizer vocab_size={vocab_size} differs from checkpoint vocab_size={saved_vocab}")
    if saved_blank is not None and saved_blank != blank_id:
        log(rank, f"  WARN: derived blank_id={blank_id} differs from checkpoint blank_id={saved_blank}")

    model = MH9MCTCModel(vocab_size=blank_id + 1, variant=saved_variant).to(device)
    # Strip a leading "module." (DDP) prefix if present.
    cleaned = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    log(rank, f"  loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        log(rank, f"    missing (first 5): {missing[:5]}")
    if unexpected[:5]:
        log(rank, f"    unexpected (first 5): {unexpected[:5]}")
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )
    model.eval()

    ctc = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    results = {}
    for split in args.splits:
        log(rank, f"evaluating {split}")
        dataset = CTCSpecDataset(
            args.data_root, [split], tokenizer,
            cmvn_path=args.cmvn_path, train_split=False,
            validate_audio=True,
        )
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                      shuffle=False, drop_last=False) if world_size > 1 else None
        loader = DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            shuffle=False, num_workers=args.workers,
            collate_fn=collate_eval, pin_memory=True,
            persistent_workers=(args.workers > 0),
            prefetch_factor=4 if args.workers > 0 else None,
        )
        m = evaluate(model, loader, tokenizer, blank_id, ctc, device, rank, world_size)
        if is_main(rank):
            results[split] = {
                "wer": m["wer"],
                "loss": m["loss"],
                "substitution_rate": m["sub_rate"],
                "insertion_rate": m["ins_rate"],
                "deletion_rate": m["del_rate"],
                "utterances": len(dataset),
                "example_hyp": m["example"][0] if m["example"] else "",
                "example_ref": m["example"][1] if m["example"] else "",
            }
            log(rank, f"  {split}: WER={100*m['wer']:.2f}% loss={m['loss']:.3f}")

    if is_main(rank):
        ckpt_dir = os.path.dirname(args.checkpoint)
        default_out = os.path.join(os.path.dirname(ckpt_dir), "eval_results.json")
        out_path = args.output or default_out
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        payload = {
            "checkpoint": args.checkpoint,
            "variant": saved_variant,
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": socket.gethostname(),
            "world_size": world_size,
            "splits": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        log(rank, f"wrote {out_path}")

    cleanup_dist()


if __name__ == "__main__":
    main()
