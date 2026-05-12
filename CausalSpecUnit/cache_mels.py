import argparse
import json
import os

import torch
from tqdm import tqdm

from CausalSpecUnit.common import TRAIN_SPLITS
from CausalSpecUnit.data import LogMelExtractor, apply_cmvn, iter_librispeech_items, load_cmvn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="dataset/datasets/librispeech/LibriSpeech")
    p.add_argument("--targets-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=TRAIN_SPLITS)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    mean, std = load_cmvn(os.path.join(args.targets_dir, "cmvn.pt"))
    extractor = LogMelExtractor()
    items = list(iter_librispeech_items(args.data_root, args.splits))

    written = skipped = failed = 0
    for item in tqdm(items, desc="cache mels"):
        split_dir = os.path.join(args.output_dir, item["split"])
        os.makedirs(split_dir, exist_ok=True)
        out_path = os.path.join(split_dir, item["uid"] + ".pt")
        if os.path.isfile(out_path) and not args.overwrite:
            skipped += 1
            continue
        try:
            mel = apply_cmvn(extractor(item["audio_path"]), mean, std).cpu().float()
            tmp_path = out_path + ".tmp"
            torch.save(mel, tmp_path)
            os.replace(tmp_path, out_path)
            written += 1
        except Exception as exc:
            failed += 1
            print(f"[warn] failed {item['uid']}: {exc}", flush=True)

    metadata = {
        "data_root": args.data_root,
        "targets_dir": args.targets_dir,
        "splits": args.splits,
        "num_items": len(items),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "feature": "cmvn_log_mel",
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Cached mels to {args.output_dir}")
    print(metadata)


if __name__ == "__main__":
    main()
