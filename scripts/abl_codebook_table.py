"""Aggregate dual-codebook ablation results into a LaTeX table.

Reads:
    <base-dir>/ft_<mode>_<hours>h/eval_results.json   for mode in {coarse, fine, both}
                                                       and hours in {10, 100}

Emits:
    1) A plain table to stdout for quick inspection.
    2) A LaTeX block formatted like the paper's tab:abl-codebook, ready to paste.

The LaTeX layout extends the single-condition table the user provided into a
two-condition view (10h, 100h) on test-clean / test-other. The bold style
follows the original: best row in each column gets \\textbf{...}.
"""

import argparse
import json
import os
import sys


MODES = ["coarse", "fine", "both"]
HOURS = [10, 100]
SPLITS = ["test-clean", "test-other"]

MODE_LABELS = {
    "coarse": r"Coarse codebook only ($K_c{=}100$)",
    "fine":   r"Fine codebook only ($K_f{=}500$)",
    "both":   r"\textbf{Joint $K_c + K_f$ (ours)}",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="outputs/causal_specunit/abl_codebook",
                   help="Directory containing ft_<mode>_<hours>h/ subdirs.")
    p.add_argument("--latex-out", default=None,
                   help="Optional path to write a .tex file. If omitted, just print.")
    return p.parse_args()


def load_results(base_dir):
    """Returns nested dict: results[mode][hours][split] = WER (float, fraction)."""
    out = {}
    missing = []
    for mode in MODES:
        out[mode] = {}
        for hours in HOURS:
            path = os.path.join(base_dir, f"ft_{mode}_{hours}h", "eval_results.json")
            if not os.path.isfile(path):
                missing.append(path)
                out[mode][hours] = None
                continue
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            out[mode][hours] = {split: payload["splits"][split]["wer"] for split in SPLITS}
    return out, missing


def fmt_pct(value):
    if value is None:
        return "—"
    return f"{value * 100:.1f}"


def bold_best(values, fmt):
    """Bold the minimum value (best WER) in a column. Ties: bold the first."""
    finite = [(i, v) for i, v in enumerate(values) if v is not None]
    if not finite:
        return [fmt(v) for v in values]
    best_idx = min(finite, key=lambda kv: kv[1])[0]
    out = []
    for i, v in enumerate(values):
        cell = fmt(v)
        if i == best_idx:
            cell = r"\textbf{" + cell + "}"
        out.append(cell)
    return out


def print_plain_table(results):
    """Quick stdout table for inspection."""
    header = f"{'mode':<10}"
    for hours in HOURS:
        for split in SPLITS:
            header += f"  {hours}h/{split:<11}"
    print(header)
    print("-" * len(header))
    for mode in MODES:
        line = f"{mode:<10}"
        for hours in HOURS:
            cells = results[mode].get(hours)
            for split in SPLITS:
                val = cells[split] if cells else None
                line += f"  {fmt_pct(val):>12}"
        print(line)


def emit_latex(results):
    """Build a LaTeX table extending the original with 10h and 100h columns."""
    # Per-column WERs: ordered as [coarse, fine, both]
    columns = {}
    for hours in HOURS:
        for split in SPLITS:
            col_key = (hours, split)
            vals = []
            for mode in MODES:
                cells = results[mode].get(hours)
                vals.append(cells[split] if cells else None)
            columns[col_key] = bold_best(vals, fmt_pct)

    # Build the row body. Each row has its mode label + 4 numeric cells.
    rows = []
    for mode_idx, mode in enumerate(MODES):
        cells = [MODE_LABELS[mode]]
        for hours in HOURS:
            for split in SPLITS:
                cells.append(columns[(hours, split)][mode_idx])
        rows.append(" & ".join(cells) + r" \\")

    latex = r"""\begin{table}[t]
  \caption{Dual-codebook ablation. SSL pretrained with each objective at 50k
    steps, then fine-tuned on \textit{train-clean-100} (full and a 10h subset)
    for 100 epochs. Test WER ($\downarrow$, \%).}
  \label{tab:abl-codebook}
  \begin{center}
    \begin{small}
      \begin{sc}
        \begin{tabular}{lcccc}
          \toprule
          \multirow{2}{*}{SSL prediction target} & \multicolumn{2}{c}{10h fine-tune} & \multicolumn{2}{c}{100h fine-tune} \\
          \cmidrule(lr){2-3} \cmidrule(lr){4-5}
                                             & clean        & other         & clean        & other         \\
          \midrule
""" + "\n".join("          " + r for r in rows) + r"""
          \bottomrule
        \end{tabular}
      \end{sc}
    \end{small}
  \end{center}
  \vskip -0.1in
\end{table}
"""
    return latex


def main():
    args = parse_args()
    results, missing = load_results(args.base_dir)
    if missing:
        print(f"[warn] {len(missing)} eval_results.json files not yet present:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        print("Will emit the table with — placeholders for those cells.\n", file=sys.stderr)

    print("=== Plain WER table (test split) ===")
    print_plain_table(results)
    print()

    latex = emit_latex(results)
    if args.latex_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.latex_out)) or ".", exist_ok=True)
        with open(args.latex_out, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"Wrote {args.latex_out}")
    print("=== LaTeX (paste into paper) ===")
    print(latex)


if __name__ == "__main__":
    main()
