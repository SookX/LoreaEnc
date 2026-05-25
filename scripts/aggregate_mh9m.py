"""Aggregate MelHuBERT-Transformer eval results across seeds.

Walks outputs/causal_specunit/mh9m_ft/${subset}/seed${seed}/eval_results.json,
groups by subset, computes mean and sample standard deviation across seeds,
and prints both a plain-text summary and a LaTeX-ready row for Table 1.

Usage:
    python scripts/aggregate_mh9m.py
    python scripts/aggregate_mh9m.py --root outputs/causal_specunit/mh9m_ft
"""

import argparse
import json
import math
import os
import sys
from glob import glob


SUBSETS = ("librilight_1h", "librilight_10h", "train-clean-100")
SPLITS = ("test-clean", "test-other")


def load_cell(path):
    """Return dict mapping split -> wer (float, in [0,1]) or None on parse failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"  ! failed to parse {path}: {exc}", file=sys.stderr)
        return None
    splits = payload.get("splits", {})
    out = {}
    for s in SPLITS:
        if s in splits and "wer" in splits[s]:
            out[s] = float(splits[s]["wer"])
        else:
            out[s] = None
    return out


def mean_std(values):
    """Sample (n-1) standard deviation. Returns (mean, std) or (mean, 0.0) for n<=1."""
    n = len(values)
    if n == 0:
        return None, None
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var)


def fmt_pct(value):
    return f"{100.0 * value:.1f}" if value is not None else "—"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/causal_specunit/mh9m_ft")
    p.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    args = p.parse_args()

    print(f"Scanning {args.root}/")
    print()

    rows = {}
    for subset in args.subsets:
        subset_dir = os.path.join(args.root, subset)
        if not os.path.isdir(subset_dir):
            print(f"  ! missing: {subset_dir}")
            continue

        cell_paths = sorted(glob(os.path.join(subset_dir, "seed*", "eval_results.json")))
        if not cell_paths:
            print(f"  {subset}: no eval_results.json files found")
            continue

        clean_wers = []
        other_wers = []
        for cp in cell_paths:
            cell = load_cell(cp)
            if cell is None:
                continue
            seed_label = os.path.basename(os.path.dirname(cp))
            tc = cell.get("test-clean")
            to = cell.get("test-other")
            print(f"  {subset}/{seed_label}: test-clean={fmt_pct(tc)}% test-other={fmt_pct(to)}%")
            if tc is not None:
                clean_wers.append(tc)
            if to is not None:
                other_wers.append(to)

        tc_mean, tc_std = mean_std(clean_wers)
        to_mean, to_std = mean_std(other_wers)
        rows[subset] = {
            "n_clean": len(clean_wers),
            "n_other": len(other_wers),
            "tc_mean": tc_mean, "tc_std": tc_std,
            "to_mean": to_mean, "to_std": to_std,
        }
        print(
            f"  -> {subset}: "
            f"test-clean {fmt_pct(tc_mean)}±{fmt_pct(tc_std).replace(' ', '')} (n={len(clean_wers)})  "
            f"test-other {fmt_pct(to_mean)}±{fmt_pct(to_std).replace(' ', '')} (n={len(other_wers)})"
        )
        print()

    if not rows:
        print("No results found.")
        return

    # --- Plain-text summary table ---
    print("=" * 72)
    print("MelHuBERT-Transformer (mh9m) — mean ± sample std over seeds (WER, %)")
    print("=" * 72)
    header = f"{'subset':18s} {'test-clean':>16s} {'test-other':>16s} {'n':>4s}"
    print(header)
    print("-" * len(header))
    for subset in args.subsets:
        if subset not in rows:
            continue
        r = rows[subset]
        tc = f"{fmt_pct(r['tc_mean'])}±{fmt_pct(r['tc_std'])}" if r["tc_mean"] is not None else "—"
        to = f"{fmt_pct(r['to_mean'])}±{fmt_pct(r['to_std'])}" if r["to_mean"] is not None else "—"
        print(f"{subset:18s} {tc:>16s} {to:>16s} {r['n_clean']:>4d}")
    print()

    # --- LaTeX-ready row matching the paper's Table 1 format ---
    print("=" * 72)
    print("LaTeX row for Table 1 (paste between iter-1 and iter-2):")
    print("=" * 72)
    cells = []
    for subset in args.subsets:
        if subset not in rows:
            cells.extend([r"\textemdash", r"\textemdash"])
            continue
        r = rows[subset]
        for mean, std in ((r["tc_mean"], r["tc_std"]), (r["to_mean"], r["to_std"])):
            if mean is None:
                cells.append(r"\textemdash")
            else:
                cells.append(f"{100*mean:.1f}{{\\scriptsize\\,$\\pm${100*std:.1f}}}")
    row = "\\quad MelHuBERT-style (mh9m) & " + " & ".join(cells) + r" \\"
    print(row)
    print()


if __name__ == "__main__":
    main()
