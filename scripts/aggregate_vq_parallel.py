"""Aggregate parallel-VQ results across {iter, budget, seed} into a
mean +/- sample-std table, ready for the paper.

Reads:
  outputs/causal_specunit/vq_smoke/parallel/<budget>/seed*/eval_results.json   (iter-1)
  outputs/causal_specunit/vq_iter2/parallel/<budget>/seed*/eval_results.json   (iter-2)

Prints:
  - per-cell raw values for sanity
  - mean +/- std grouped by (iter, budget)
  - a LaTeX row pair ready to paste into Table 1

Usage:
    python scripts/aggregate_vq_parallel.py
"""

import json
import math
import os
import sys
from glob import glob


ROOTS = {
    "iter1": "outputs/causal_specunit/vq_smoke/parallel",
    "iter2": "outputs/causal_specunit/vq_iter2/parallel",
}
BUDGETS = ("librilight_1h", "librilight_10h", "train-clean-100")
SPLITS = ("test-clean", "test-other")


def load_wer(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"  ! failed to parse {path}: {exc}", file=sys.stderr)
        return {}
    splits = payload.get("splits", {})
    return {s: float(splits[s]["wer"]) for s in SPLITS if s in splits and "wer" in splits[s]}


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var)


def main() -> None:
    rows: dict = {}

    for iter_label, root in ROOTS.items():
        for budget in BUDGETS:
            cells = sorted(glob(os.path.join(root, budget, "seed*", "eval_results.json")))
            clean, other = [], []
            for cp in cells:
                wer = load_wer(cp)
                seed_label = os.path.basename(os.path.dirname(cp))
                tc = wer.get("test-clean")
                to = wer.get("test-other")
                if tc is not None:
                    clean.append(tc)
                if to is not None:
                    other.append(to)
                print(f"  {iter_label} {budget} {seed_label}: "
                      f"clean={100*tc:.2f}% other={100*to:.2f}%" if (tc and to)
                      else f"  {iter_label} {budget} {seed_label}: incomplete")
            tc_mean, tc_std = mean_std(clean)
            to_mean, to_std = mean_std(other)
            rows[(iter_label, budget)] = {
                "n": len(clean),
                "tc_mean": tc_mean, "tc_std": tc_std,
                "to_mean": to_mean, "to_std": to_std,
            }

    # Summary table
    print()
    print("=" * 80)
    print("Parallel-VQ summary (WER %, mean +/- sample-std over seeds)")
    print("=" * 80)
    header = f"{'iter':6s} {'budget':18s} {'test-clean':>16s} {'test-other':>16s} {'n':>4s}"
    print(header)
    print("-" * len(header))
    for iter_label in ("iter1", "iter2"):
        for budget in BUDGETS:
            r = rows.get((iter_label, budget))
            if r is None or r["n"] == 0:
                print(f"{iter_label:6s} {budget:18s} {'-':>16s} {'-':>16s} {'0':>4s}")
                continue
            tc = f"{100*r['tc_mean']:.2f}+-{100*r['tc_std']:.2f}"
            to = f"{100*r['to_mean']:.2f}+-{100*r['to_std']:.2f}"
            print(f"{iter_label:6s} {budget:18s} {tc:>16s} {to:>16s} {r['n']:>4d}")

    # LaTeX rows for the paper
    print()
    print("=" * 80)
    print("LaTeX rows for Table 1 (paste between iter-1 dual-kmeans and iter-2 dual-kmeans):")
    print("=" * 80)

    def cell_str(mean: float, std: float) -> str:
        if math.isnan(mean):
            return r"\textemdash"
        return f"{100*mean:.1f}{{\\scriptsize\\,$\\pm${100*std:.1f}}}"

    for iter_label, row_label in (
        ("iter1", r"\quad + SSL iter-1 (parallel-VQ)"),
        ("iter2", r"\quad \textbf{+ SSL iter-2 (parallel-VQ, ours)}"),
    ):
        cells = []
        for budget in BUDGETS:
            r = rows.get((iter_label, budget), {})
            cells.append(cell_str(r.get("tc_mean", float("nan")), r.get("tc_std", 0.0)))
            cells.append(cell_str(r.get("to_mean", float("nan")), r.get("to_std", 0.0)))
        print(row_label + " & " + " & ".join(cells) + r" \\")
    print()


if __name__ == "__main__":
    main()
