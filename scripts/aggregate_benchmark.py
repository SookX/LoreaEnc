"""Aggregate the 1h/10h/100h benchmark across {subset, condition, seed} into a
mean +/- sample-std table, so KD-distillation rows sit next to our SSL rows.

Reads the tree written by slurm/causal_specunit/10_benchmark_1h_10h_100h_3seeds.sh:
  <root>/<subset>/<condition>_seed<N>/eval_results.json

Conditions are auto-discovered, so distill_hubert / distill_wav2vec2 appear as
soon as those cells finish. Because every condition is fine-tuned by that one
script with an identical recipe, the rows are directly comparable.

Prints:
  - per-cell raw WER for sanity (and which cells are still missing)
  - mean +/- std grouped by (subset, condition)
  - a LaTeX row per condition, ready to paste into the paper table

Usage:
    python scripts/aggregate_benchmark.py
    python scripts/aggregate_benchmark.py --root outputs/causal_specunit/benchmark_1h_10h_100h_4gpu
    python scripts/aggregate_benchmark.py --subsets librilight_1h
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from glob import glob

DEFAULT_ROOT = "outputs/causal_specunit/benchmark_1h_10h_100h_4gpu"
SUBSET_ORDER = ("librilight_1h", "librilight_10h", "train-clean-100")
SPLITS = ("test-clean", "test-other")

# Preferred display order; anything else is appended alphabetically.
CONDITION_ORDER = ("scratch", "iter1", "iter2", "distill_hubert", "distill_wav2vec2")
CONDITION_LABEL = {
    "scratch": r"\quad scratch (no pretraining)",
    "iter1": r"\quad + SSL iter-1 (ours)",
    "iter2": r"\quad \textbf{+ SSL iter-2 (ours)}",
    "distill_hubert": r"\quad + KD from HuBERT Base",
    "distill_wav2vec2": r"\quad + KD from wav2vec2 Base",
}


def load_wer(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"  ! failed to parse {path}: {exc}", file=sys.stderr)
        return {}
    splits = payload.get("splits", {})
    return {s: float(splits[s]["wer"]) for s in SPLITS if s in splits and "wer" in splits[s]}


def mean_std(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var)


def parse_cell(dirname):
    """'distill_hubert_seed42' -> ('distill_hubert', '42'); None if unparseable."""
    if "_seed" not in dirname:
        return None
    condition, seed = dirname.rsplit("_seed", 1)
    return (condition, seed) if condition and seed else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="Limit to these subsets (default: all found).")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"No such results root: {args.root}", file=sys.stderr)
        return 1

    # cells[(subset, condition)][seed] = {split: wer}
    cells = defaultdict(dict)
    found_subsets, found_conditions = set(), set()

    for cell_path in sorted(glob(os.path.join(args.root, "*", "*_seed*"))):
        if not os.path.isdir(cell_path):
            continue
        subset = os.path.basename(os.path.dirname(cell_path))
        parsed = parse_cell(os.path.basename(cell_path))
        if parsed is None:
            continue
        condition, seed = parsed
        if args.subsets and subset not in args.subsets:
            continue
        found_subsets.add(subset)
        found_conditions.add(condition)
        results_path = os.path.join(cell_path, "eval_results.json")
        cells[(subset, condition)][seed] = load_wer(results_path) if os.path.isfile(results_path) else None

    if not cells:
        print(f"No benchmark cells found under {args.root}", file=sys.stderr)
        return 1

    subsets = [s for s in SUBSET_ORDER if s in found_subsets] + sorted(found_subsets - set(SUBSET_ORDER))
    conditions = [c for c in CONDITION_ORDER if c in found_conditions] + sorted(found_conditions - set(CONDITION_ORDER))

    # ---- per-cell raw values -------------------------------------------------
    print("=" * 84)
    print(f"Per-cell WER (root: {args.root})")
    print("=" * 84)
    for subset in subsets:
        for condition in conditions:
            seeds = cells.get((subset, condition))
            if not seeds:
                continue
            for seed in sorted(seeds):
                wer = seeds[seed]
                if not wer:
                    print(f"  {subset:16s} {condition:18s} seed{seed:<4s} PENDING (no eval_results.json)")
                    continue
                tc, to = wer.get("test-clean"), wer.get("test-other")
                tc_s = f"{100*tc:.2f}%" if tc is not None else "-"
                to_s = f"{100*to:.2f}%" if to is not None else "-"
                print(f"  {subset:16s} {condition:18s} seed{seed:<4s} clean={tc_s:>8s} other={to_s:>8s}")

    # ---- summary -------------------------------------------------------------
    stats = {}
    print()
    print("=" * 84)
    print("Summary (WER %, mean +/- sample-std over seeds)")
    print("=" * 84)
    header = f"{'subset':16s} {'condition':18s} {'test-clean':>16s} {'test-other':>16s} {'n':>3s}"
    print(header)
    print("-" * len(header))
    for subset in subsets:
        for condition in conditions:
            seeds = cells.get((subset, condition))
            if not seeds:
                continue
            clean = [w["test-clean"] for w in seeds.values() if w and "test-clean" in w]
            other = [w["test-other"] for w in seeds.values() if w and "test-other" in w]
            tc_mean, tc_std = mean_std(clean)
            to_mean, to_std = mean_std(other)
            stats[(subset, condition)] = (tc_mean, tc_std, to_mean, to_std, len(clean))
            if not clean:
                print(f"{subset:16s} {condition:18s} {'-':>16s} {'-':>16s} {0:>3d}")
                continue
            tc = f"{100*tc_mean:.2f}+-{100*tc_std:.2f}"
            to = f"{100*to_mean:.2f}+-{100*to_std:.2f}"
            print(f"{subset:16s} {condition:18s} {tc:>16s} {to:>16s} {len(clean):>3d}")
        print()

    # ---- LaTeX ---------------------------------------------------------------
    def cell_str(mean, std):
        if math.isnan(mean):
            return r"\textemdash"
        return f"{100*mean:.1f}{{\\scriptsize\\,$\\pm${100*std:.1f}}}"

    print("=" * 84)
    print(f"LaTeX rows (columns: {' | '.join(f'{s} clean/other' for s in subsets)})")
    print("=" * 84)
    for condition in conditions:
        row = []
        for subset in subsets:
            tc_mean, tc_std, to_mean, to_std, _ = stats.get(
                (subset, condition), (float("nan"), 0.0, float("nan"), 0.0, 0))
            row.append(cell_str(tc_mean, tc_std))
            row.append(cell_str(to_mean, to_std))
        label = CONDITION_LABEL.get(condition, rf"\quad {condition}")
        print(label + " & " + " & ".join(row) + r" \\")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
