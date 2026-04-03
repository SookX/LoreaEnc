"""
dataset/splits.py
─────────────────
Creates reproducible labeled-data manifests for the semi-supervised benchmark.

Generates three JSON files under dataset/splits/:
    1h.json   — ~1 hour   of labeled LibriSpeech utterances
    10h.json  — ~10 hours of labeled LibriSpeech utterances
    100h.json — full train-clean-100 (~100 hours)

Both Lorea and the wav2vec2/HuBERT baselines must use these exact manifests
so comparisons are fair.  Run once before any training:

    python dataset/splits.py --data_root ./dataset --seed 42

Each manifest is a JSON list of {"path": "...", "transcript": "...", "duration_s": ...}.
"""

import os
import sys
import json
import random
import argparse
import logging

import torchaudio

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Directory walker
# ─────────────────────────────────────────────────────────────────────────────

def _walk_split(data_root: str, split: str):
    """Yield (path, transcript) pairs for one LibriSpeech split."""
    split_dir = os.path.join(data_root, split)
    for speaker in sorted(os.listdir(split_dir)):
        speaker_dir = os.path.join(split_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        for section in sorted(os.listdir(speaker_dir)):
            section_dir = os.path.join(speaker_dir, section)
            if not os.path.isdir(section_dir):
                continue
            txt_files = [f for f in os.listdir(section_dir) if f.endswith(".txt")]
            if not txt_files:
                continue
            with open(os.path.join(section_dir, txt_files[0])) as f:
                for line in f:
                    parts = line.split()
                    uid = parts[0]
                    transcript = " ".join(parts[1:]).strip()
                    path = os.path.join(section_dir, uid + ".flac")
                    yield path, transcript


def _get_duration(path: str) -> float:
    """Return audio duration in seconds without loading samples."""
    info = torchaudio.info(path)
    return info.num_frames / info.sample_rate


# ─────────────────────────────────────────────────────────────────────────────
# Subset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_subset(entries, target_hours: float, seed: int):
    """
    Randomly sample entries (shuffled with `seed`) until `target_hours` is reached.
    entries: list of {"path", "transcript", "duration_s"}
    Returns a list of the sampled entries.
    """
    rng = random.Random(seed)
    shuffled = entries[:]
    rng.shuffle(shuffled)

    target_s = target_hours * 3600.0
    total = 0.0
    subset = []
    for e in shuffled:
        if total >= target_s:
            break
        subset.append(e)
        total += e["duration_s"]

    logger.info(
        f"  {target_hours}h subset: {len(subset)} utterances, "
        f"{total/3600:.2f} h actual"
    )
    return subset


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    p = argparse.ArgumentParser(description="Create reproducible labeled-data splits")
    p.add_argument("--data_root", default="./dataset")
    p.add_argument("--split",     default="train-clean-100",
                   help="LibriSpeech split to draw labeled subsets from")
    p.add_argument("--out_dir",   default="./dataset/splits")
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load all utterances with durations ────────────────────────────────
    logger.info(f"Scanning {args.split} …")
    entries = []
    for path, transcript in _walk_split(args.data_root, args.split):
        dur = _get_duration(path)
        entries.append({"path": path, "transcript": transcript, "duration_s": dur})

    total_h = sum(e["duration_s"] for e in entries) / 3600
    logger.info(f"Found {len(entries)} utterances, {total_h:.1f} h total")

    # ── 100 h = full split ────────────────────────────────────────────────
    out_100h = os.path.join(args.out_dir, "100h.json")
    with open(out_100h, "w") as f:
        json.dump(entries, f, indent=2)
    logger.info(f"Saved 100h.json ({len(entries)} utterances, {total_h:.1f} h)")

    # ── 10 h subset ───────────────────────────────────────────────────────
    subset_10h = build_subset(entries, target_hours=10.0, seed=args.seed)
    out_10h = os.path.join(args.out_dir, "10h.json")
    with open(out_10h, "w") as f:
        json.dump(subset_10h, f, indent=2)
    logger.info(f"Saved 10h.json")

    # ── 1 h subset ────────────────────────────────────────────────────────
    subset_1h = build_subset(entries, target_hours=1.0, seed=args.seed)
    out_1h = os.path.join(args.out_dir, "1h.json")
    with open(out_1h, "w") as f:
        json.dump(subset_1h, f, indent=2)
    logger.info(f"Saved 1h.json")

    logger.info("Done. Use these manifests with --train_manifest for both baselines and Lorea.")


if __name__ == "__main__":
    main()
