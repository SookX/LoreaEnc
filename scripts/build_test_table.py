"""Aggregate test-set WER from outputs/causal_specunit/test_table/ into a
plain-text table and a LaTeX block matching Table 1 of the paper.

Expected directory layout (matches 06_test_table.sh + 07_eval_960h_existing.sh):
    <base-dir>/<cond>_<hours>h/eval_results.json
where cond in {scratch, iter1, iter2} and hours in {10, 100, 960}.

Each eval_results.json must have:
    {"splits": {"test-clean": {"wer": float, ...},
                "test-other": {"wer": float, ...}}, ...}

Usage:
    python scripts/build_test_table.py
    python scripts/build_test_table.py --base-dir outputs/causal_specunit/test_table
    python scripts/build_test_table.py --latex-out paper_table.tex
"""

import argparse
import json
import os
import sys


CONDS = ["scratch", "iter1", "iter2"]
HOURS = [10, 100, 960]
SPLITS = ["test-clean", "test-other"]

ROW_LABELS = {
    "scratch": r"SqueezeFormer-XS (scratch)",
    "iter1":   r"\quad + SSL (iter-1)",
    "iter2":   r"\quad \textbf{+ SSL (iter-2, ours)}",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="outputs/causal_specunit/test_table",
                   help="Directory containing <cond>_<hours>h/ subdirs.")
    p.add_argument("--latex-out", default=None,
                   help="Optional path to write the LaTeX block.")
    return p.parse_args()


def load_results(base_dir):
    """Returns nested dict: results[cond][hours][split] = WER (fraction)."""
    out = {}
    missing = []
    for cond in CONDS:
        out[cond] = {}
        for hours in HOURS:
            path = os.path.join(base_dir, f"{cond}_{hours}h", "eval_results.json")
            if not os.path.isfile(path):
                missing.append(path)
                out[cond][hours] = None
                continue
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            try:
                out[cond][hours] = {sp: payload["splits"][sp]["wer"] for sp in SPLITS}
            except KeyError as exc:
                print(f"[warn] {path}: missing key {exc}", file=sys.stderr)
                out[cond][hours] = None
    return out, missing


def fmt_pct(value):
    if value is None:
        return "--"
    return f"{value * 100:.2f}"


def bold_best(values, fmt):
    """Bold the minimum value (best WER) in a column."""
    finite = [(i, v) for i, v in enumerate(values) if v is not None]
    if not finite:
        return [fmt(v) for v in values]
    best_idx = min(finite, key=lambda kv: kv[1])[0]
    out = []
    for i, v in enumerate(values):
        cell = fmt(v)
        if i == best_idx and v is not None:
            cell = r"\textbf{" + cell + "}"
        out.append(cell)
    return out


def print_plain(results):
    """Per-row, three-data-budget plain text."""
    print("test-set WER per condition (%):")
    print()
    header = f"{'condition':<14}" + "".join(
        f"  {str(h)+'h':<8}{split[5:]:<8}" if split == "test-clean" else f"{split[5:]:<8}"
        for h in HOURS for split in SPLITS
    )
    # Cleaner header: condition  10h-clean  10h-other  100h-clean ...
    cols = []
    for h in HOURS:
        for split in SPLITS:
            cols.append(f"{h}h/{split.replace('test-', '')}")
    print(f"{'condition':<14}  " + "  ".join(f"{c:>11}" for c in cols))
    print("-" * (14 + 2 + 13 * len(cols)))
    for cond in CONDS:
        row_cells = []
        for h in HOURS:
            cells = results[cond].get(h)
            for split in SPLITS:
                val = cells[split] if cells else None
                row_cells.append(fmt_pct(val))
        print(f"{cond:<14}  " + "  ".join(f"{c:>11}" for c in row_cells))


def emit_latex(results):
    """Build a LaTeX table matching the paper's Table 1."""
    # Per-column WERs ordered as [scratch, iter1, iter2]
    columns = {}
    for h in HOURS:
        for split in SPLITS:
            vals = [results[c].get(h, {}).get(split) if results[c].get(h) else None
                    for c in CONDS]
            columns[(h, split)] = bold_best(vals, fmt_pct)

    rows = []
    for cond_idx, cond in enumerate(CONDS):
        cells = [ROW_LABELS[cond]]
        for h in HOURS:
            for split in SPLITS:
                cells.append(columns[(h, split)][cond_idx])
        rows.append(" & ".join(cells) + r" \\")

    latex = r"""\begin{table*}[t]
  \caption{LibriSpeech WER ($\downarrow$, \%) on \textit{test-clean} /
    \textit{test-other} at three labeled-data budgets, evaluated with
    greedy CTC decoding (no external language model). All rows use the
    same SqueezeFormer-XS architecture and the same 150-epoch CTC recipe;
    rows differ only in encoder initialization. Iter-1 clusters log-mel
    chunks directly; iter-2 re-clusters the iter-1 encoder's hidden
    states. Best per column in \textbf{bold}.}
  \label{tab:main}
  \begin{center}
    \begin{small}
      \begin{sc}
        \begin{tabular}{lcccccc}
          \toprule
                                              & \multicolumn{2}{c}{10h} & \multicolumn{2}{c}{100h} & \multicolumn{2}{c}{960h} \\
          \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
          Method                              & clean & other & clean & other & clean & other \\
          \midrule
""" + "\n".join("          " + r for r in rows) + r"""
          \bottomrule
        \end{tabular}
      \end{sc}
    \end{small}
  \end{center}
  \vskip -0.1in
\end{table*}
"""
    return latex


def main():
    args = parse_args()
    results, missing = load_results(args.base_dir)
    if missing:
        print(f"[warn] {len(missing)} eval_results.json not found "
              f"(table cells will show --):", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        print("", file=sys.stderr)

    print("=== Plain test-set WER table ===")
    print_plain(results)
    print()

    latex = emit_latex(results)
    if args.latex_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.latex_out)) or ".", exist_ok=True)
        with open(args.latex_out, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"Wrote LaTeX to {args.latex_out}", file=sys.stderr)
        print()
    print("=== LaTeX (paste into paper, replaces current Table 1) ===")
    print(latex)


if __name__ == "__main__":
    main()
