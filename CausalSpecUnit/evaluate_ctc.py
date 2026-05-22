"""Post-hoc evaluation of a trained CTC checkpoint on arbitrary LibriSpeech splits.

Used by the ablation pipeline to score the same checkpoint on test-clean and
test-other (the training-time eval is dev-other only).

Output: writes a JSON file `eval_results.json` in the checkpoint directory
with keys per split. Format:

  {
    "checkpoint": "outputs/.../ft_abl_both_100h/checkpoint_best/checkpoint.pt",
    "splits": {
      "test-clean": {"wer": 0.082, "cer": 0.031, "utterances": 2620, ...},
      "test-other": {"wer": 0.198, "cer": 0.092, "utterances": 2939, ...}
    }
  }
"""

import argparse
import json
import os
import socket
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from SqueezeFormer.train import build_tokenizer
from CausalSpecUnit.common import (
    barrier,
    cleanup_distributed,
    is_main_process,
    print0,
    setup_distributed,
    strip_state_prefixes,
)
from CausalSpecUnit.data import CTCSpecDataset, collate_eval
from CausalSpecUnit.model import CausalSpecUnitCTC
from CausalSpecUnit.train_ctc import evaluate, unwrap_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a checkpoint directory or checkpoint.pt file.")
    p.add_argument("--data-root", type=str, default="dataset/datasets/librispeech/LibriSpeech")
    p.add_argument("--cmvn-path", type=str, default="outputs/causal_specunit/targets_960h_c8/cmvn.pt")
    p.add_argument("--tokenizer-path", type=str, default="dataset/bpe128.model")
    p.add_argument("--variant", type=str, default="xs", choices=["xs", "s", "sm", "m", "ml", "l"])
    p.add_argument("--splits", nargs="+", default=["test-clean", "test-other"],
                   help="LibriSpeech splits to evaluate on.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=str, default=None,
                   help="Where to write eval_results.json. Default: alongside the checkpoint.")
    p.add_argument("--inter-ctc-layers", type=int, nargs="*", default=None,
                   help="Must match what the checkpoint was trained with. The aux heads in the checkpoint "
                        "are unused at eval (we only call the main head), but constructing the model "
                        "without them when they exist in the state_dict triggers strict-load mismatches.")
    return p.parse_args()


def resolve_checkpoint(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidate = os.path.join(path, "checkpoint.pt")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"No checkpoint at {path}")


def detect_inter_ctc_layers(state_dict):
    """Pull inter_ctc layer indices from the model state_dict.

    Saved keys look like ``model.inter_ctc_heads.<layer_idx>.weight``.
    Returning the indices keeps the model build symmetric with training,
    so load_state_dict succeeds without surprises."""
    layers = set()
    for key in state_dict.keys():
        if "inter_ctc_heads." in key:
            after = key.split("inter_ctc_heads.", 1)[1]
            try:
                layers.add(int(after.split(".", 1)[0]))
            except (IndexError, ValueError):
                continue
    return sorted(layers)


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    ckpt_path = resolve_checkpoint(args.checkpoint)
    state = torch.load(ckpt_path, map_location="cpu")
    model_state = strip_state_prefixes(state["model"] if "model" in state else state)

    inter_ctc_layers = args.inter_ctc_layers
    if inter_ctc_layers is None:
        inter_ctc_layers = detect_inter_ctc_layers(model_state)

    tokenizer = build_tokenizer(args.tokenizer_path)
    blank_id = tokenizer.pad_token_id

    model = CausalSpecUnitCTC(
        vocab_size=tokenizer.vocab_size,
        variant=args.variant,
        inter_ctc_layers=inter_ctc_layers or None,
    )
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if is_main_process(rank):
        print0(rank, f"Loaded checkpoint {ckpt_path}")
        print0(rank, f"  inter_ctc_layers detected={inter_ctc_layers}")
        print0(rank, f"  load missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print0(rank, f"  missing sample: {missing[:5]}")
        if unexpected:
            print0(rank, f"  unexpected sample: {unexpected[:5]}")
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()
    ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    results = {}
    for split in args.splits:
        dataset = CTCSpecDataset(
            args.data_root,
            [split],
            tokenizer,
            cmvn_path=args.cmvn_path,
            train_split=False,
            validate_audio=True,
        )
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        worker_kwargs = {"persistent_workers": True, "prefetch_factor": 2} if args.workers > 0 else {}
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.workers,
            collate_fn=collate_eval,
            pin_memory=True,
            **worker_kwargs,
        )
        print0(rank, f"Evaluating {split} | utterances={len(dataset)}")
        metrics = evaluate(model, loader, tokenizer, blank_id, ctc_loss, device, rank, show_progress=is_main_process(rank))
        if is_main_process(rank):
            results[split] = {
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "loss": metrics["loss"],
                "substitution_rate": metrics["substitution_rate"],
                "insertion_rate": metrics["insertion_rate"],
                "deletion_rate": metrics["deletion_rate"],
                "utterances": metrics["eval_utterances"],
                "example_hyp": metrics["example"][0] if metrics["example"] else "",
                "example_ref": metrics["example"][1] if metrics["example"] else "",
            }
            print0(rank, f"  {split}: WER={metrics['wer']:.4%} CER={metrics['cer']:.4%}")
        barrier()

    if is_main_process(rank):
        output_path = args.output or os.path.join(
            os.path.dirname(ckpt_path) if os.path.isfile(args.checkpoint) else args.checkpoint,
            "eval_results.json",
        )
        payload = {
            "checkpoint": ckpt_path,
            "variant": args.variant,
            "inter_ctc_layers": inter_ctc_layers,
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": socket.gethostname(),
            "world_size": world_size,
            "splits": results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print0(rank, f"Wrote {output_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
