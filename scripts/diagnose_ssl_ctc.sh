#!/usr/bin/env bash
# Print a compact but information-rich report for one SSL pretrain run and
# one downstream CTC fine-tune run.
#
# Usage:
#   scripts/diagnose_ssl_ctc.sh \
#     outputs/causal_specunit/ssl_m95_iter1_50k \
#     outputs/causal_specunit/m95_smoke/iter1/librilight_1h/seed42_ssl50k
#
# Optional:
#   TOP_N=12 scripts/diagnose_ssl_ctc.sh SSL_DIR FT_DIR

set -euo pipefail

SSL_DIR="${1:-outputs/causal_specunit/ssl_m95_iter1_50k}"
FT_DIR="${2:-outputs/causal_specunit/m95_smoke/iter1/librilight_1h/seed42_ssl50k}"
TOP_N="${TOP_N:-10}"

python3 - "$SSL_DIR" "$FT_DIR" "$TOP_N" <<'PY'
import json
import math
import os
import sys
from pathlib import Path


ssl_dir = Path(sys.argv[1])
ft_dir = Path(sys.argv[2])
top_n = int(sys.argv[3])


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def read_jsonl(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    rows.append({"event": "parse_error", "lineno": lineno, "error": str(exc)})
    except FileNotFoundError:
        return []
    return rows


def fmt(v, digits=4, missing="-"):
    if v is None:
        return missing
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        return f"{v:.{digits}f}"
    return str(v)


def pct(v, digits=2, missing="-"):
    if v is None:
        return missing
    return f"{100.0 * float(v):.{digits}f}%"


def val(row, key, default=None):
    return row.get(key, default) if row else default


def checkpoint_steps(path):
    if not path.is_dir():
        return []
    steps = []
    for child in path.iterdir():
        if child.name.startswith("checkpoint_step"):
            suffix = child.name.replace("checkpoint_step", "")
            if suffix.isdigit():
                steps.append(int(suffix))
    return sorted(steps)


def print_kv(title, pairs):
    print(f"\n== {title} ==")
    for key, value in pairs:
        print(f"{key:34s} {value}")


def print_table(headers, rows):
    if not rows:
        print("(none)")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def slope(rows, key, n=5):
    usable = [r for r in rows if r.get(key) is not None and r.get("optimizer_step") is not None]
    if len(usable) < 2:
        return None
    tail = usable[-n:]
    if len(tail) < 2:
        return None
    dy = float(tail[-1][key]) - float(tail[0][key])
    dx = float(tail[-1]["optimizer_step"]) - float(tail[0]["optimizer_step"])
    if dx == 0:
        return None
    return dy / dx * 1000.0


print(f"SSL_DIR: {ssl_dir}")
print(f"FT_DIR:  {ft_dir}")

ssl_info = read_json(ssl_dir / "ssl_run_info.json")
ssl_rows = read_jsonl(ssl_dir / "ssl_metrics.jsonl")
ssl_epochs = [r for r in ssl_rows if r.get("event") == "epoch_end"]
ssl_steps = [r for r in ssl_rows if r.get("event") == "train_step"]
ssl_parse_errors = [r for r in ssl_rows if r.get("event") == "parse_error"]

if ssl_info:
    args = ssl_info.get("args", {})
    params = ssl_info.get("parameter_counts", {})
    target_stats = ssl_info.get("target_length_stats", {})
    print_kv("SSL Config", [
        ("variant", args.get("variant")),
        ("lr", args.get("lr")),
        ("warmup_epochs", args.get("warmup_epochs")),
        ("peak_epochs", args.get("peak_epochs")),
        ("warmup_steps", args.get("warmup_steps")),
        ("peak_steps", args.get("peak_steps")),
        ("max_steps", args.get("max_steps")),
        ("batch_size", args.get("batch_size")),
        ("grad_accum_steps", args.get("grad_accum_steps")),
        ("world_size", ssl_info.get("world_size")),
        ("effective_utterance_batch", ssl_info.get("effective_utterance_batch")),
        ("est_target_tokens_per_step", fmt(ssl_info.get("estimated_target_tokens_per_optimizer_step"), 1)),
        ("mask_prob", args.get("mask_prob")),
        ("mask_length", args.get("mask_length")),
        ("codebook_mode", args.get("codebook_mode")),
        ("params_total", params.get("total")),
        ("params_encoder", params.get("encoder")),
        ("target_tokens_mean", fmt(target_stats.get("target_tokens_mean"), 1)),
    ])
else:
    print("\n== SSL Config ==\nmissing ssl_run_info.json")

if ssl_epochs:
    best_ssl = min(ssl_epochs, key=lambda r: r.get("loss", float("inf")))
    last_ssl = ssl_epochs[-1]
    first_ssl = ssl_epochs[0]
    unsafe = [r for r in ssl_steps if r.get("grad_is_safe") is False]
    ckpts = checkpoint_steps(ssl_dir)
    print_kv("SSL Summary", [
        ("epochs_logged", len(ssl_epochs)),
        ("first_epoch_loss", fmt(first_ssl.get("loss"))),
        ("last_epoch", last_ssl.get("epoch")),
        ("last_optimizer_step", last_ssl.get("optimizer_step")),
        ("last_loss", fmt(last_ssl.get("loss"))),
        ("last_c100", fmt(last_ssl.get("c100"))),
        ("last_c500", fmt(last_ssl.get("c500"))),
        ("last_masked_fraction", pct(last_ssl.get("masked_fraction"))),
        ("best_ssl_epoch", best_ssl.get("epoch")),
        ("best_ssl_step", best_ssl.get("optimizer_step")),
        ("best_ssl_loss", fmt(best_ssl.get("loss"))),
        ("tail_loss_slope_per_1k_steps", fmt(slope(ssl_epochs, "loss"), 5)),
        ("unsafe_grad_records", len(unsafe)),
        ("checkpoints", ", ".join(str(x) for x in ckpts[-8:]) if ckpts else "-"),
        ("parse_errors", len(ssl_parse_errors)),
    ])
    print("\n== SSL Epoch Curve Tail ==")
    tail_rows = ssl_epochs[-top_n:]
    print_table(
        ["epoch", "step", "loss", "c100", "c500", "mask", "lr", "skipped"],
        [
            [
                r.get("epoch"),
                r.get("optimizer_step"),
                fmt(r.get("loss")),
                fmt(r.get("c100")),
                fmt(r.get("c500")),
                pct(r.get("masked_fraction")),
                fmt(r.get("lr"), 2),
                r.get("skipped_steps_epoch", 0),
            ]
            for r in tail_rows
        ],
    )
else:
    print("\n== SSL Summary ==\nmissing or empty ssl_metrics.jsonl")

ctc_info = read_json(ft_dir / "ctc_run_info.json")
ctc_rows = read_jsonl(ft_dir / "ctc_metrics.jsonl")
ctc_epochs = [r for r in ctc_rows if r.get("event") == "epoch_end" and r.get("wer") is not None]
ctc_parse_errors = [r for r in ctc_rows if r.get("event") == "parse_error"]

if ctc_info:
    args = ctc_info.get("args", {})
    params = ctc_info.get("parameter_counts", {})
    print_kv("CTC Config", [
        ("variant", args.get("variant")),
        ("ssl_checkpoint", args.get("ssl_checkpoint")),
        ("epochs", args.get("epochs")),
        ("encoder_lr", args.get("encoder_lr")),
        ("head_lr", args.get("head_lr")),
        ("base_lr", args.get("lr")),
        ("warmup_epochs", args.get("warmup_epochs")),
        ("peak_epochs", args.get("peak_epochs")),
        ("freeze_encoder_epochs", args.get("freeze_encoder_epochs")),
        ("encoder_rewarmup_epochs", args.get("encoder_rewarmup_epochs")),
        ("batch_size", args.get("batch_size")),
        ("grad_accum_steps", args.get("grad_accum_steps")),
        ("world_size", ctc_info.get("world_size")),
        ("effective_batch", ctc_info.get("effective_batch")),
        ("train_utterances", ctc_info.get("train_utterances")),
        ("train_audio_hours", ctc_info.get("train_audio_hours")),
        ("dev_utterances", ctc_info.get("dev_utterances")),
        ("params_total", params.get("total")),
        ("params_encoder", params.get("encoder")),
    ])
else:
    print("\n== CTC Config ==\nmissing ctc_run_info.json")

if ctc_epochs:
    best = min(ctc_epochs, key=lambda r: r.get("wer", float("inf")))
    last = ctc_epochs[-1]
    min_train = min(ctc_epochs, key=lambda r: r.get("train_loss", float("inf")))
    print_kv("CTC Summary", [
        ("epochs_logged", len(ctc_epochs)),
        ("last_epoch", last.get("epoch")),
        ("last_dev_wer", pct(last.get("wer"))),
        ("last_dev_cer", pct(last.get("cer"))),
        ("last_train_loss", fmt(last.get("train_loss"))),
        ("best_epoch", best.get("epoch")),
        ("best_dev_wer", pct(best.get("wer"))),
        ("best_dev_cer", pct(best.get("cer"))),
        ("best_hyp_words_per_ref", fmt(best.get("hyp_words_per_ref_word"), 3)),
        ("best_deletion_rate", pct(best.get("deletion_rate"))),
        ("best_insertion_rate", pct(best.get("insertion_rate"))),
        ("best_substitution_rate", pct(best.get("substitution_rate"))),
        ("min_train_loss_epoch", min_train.get("epoch")),
        ("min_train_loss", fmt(min_train.get("train_loss"))),
        ("tail_wer_slope_per_epoch", fmt((last.get("wer", 0) - ctc_epochs[-min(10, len(ctc_epochs))].get("wer", 0)) / max(1, min(10, len(ctc_epochs)) - 1), 5)),
        ("parse_errors", len(ctc_parse_errors)),
    ])

    best_idx = ctc_epochs.index(best)
    start = max(0, best_idx - top_n // 2)
    end = min(len(ctc_epochs), start + top_n)
    print("\n== CTC Around Best Dev WER ==")
    print_table(
        ["mark", "epoch", "dev_wer", "dev_cer", "train_loss", "lr", "clip", "hyp/ref", "del", "ins"],
        [
            [
                "*" if r is best else "",
                r.get("epoch"),
                pct(r.get("wer")),
                pct(r.get("cer")),
                fmt(r.get("train_loss")),
                fmt(r.get("lr"), 2),
                pct(r.get("clip_fraction")),
                fmt(r.get("hyp_words_per_ref_word"), 3),
                pct(r.get("deletion_rate")),
                pct(r.get("insertion_rate")),
            ]
            for r in ctc_epochs[start:end]
        ],
    )

    print("\n== CTC Curve Every 10 Epochs ==")
    every = [r for r in ctc_epochs if r.get("epoch") == 1 or r.get("epoch") % 10 == 0 or r is best or r is last]
    seen = set()
    filtered = []
    for r in every:
        epoch = r.get("epoch")
        if epoch in seen:
            continue
        seen.add(epoch)
        filtered.append(r)
    print_table(
        ["epoch", "dev_wer", "dev_cer", "train_loss", "lr", "specaug", "hyp/ref"],
        [
            [
                r.get("epoch"),
                pct(r.get("wer")),
                pct(r.get("cer")),
                fmt(r.get("train_loss")),
                fmt(r.get("lr"), 2),
                r.get("specaug_enabled"),
                fmt(r.get("hyp_words_per_ref_word"), 3),
            ]
            for r in filtered
        ],
    )
else:
    print("\n== CTC Summary ==\nmissing or empty ctc_metrics.jsonl")

eval_results = read_json(ft_dir / "eval_results.json")
if eval_results:
    print("\n== Test Results ==")
    splits = eval_results.get("splits", {})
    print_table(
        ["split", "wer", "cer", "del", "ins", "sub", "loss", "utts"],
        [
            [
                name,
                pct(row.get("wer")),
                pct(row.get("cer")),
                pct(row.get("deletion_rate")),
                pct(row.get("insertion_rate")),
                pct(row.get("substitution_rate")),
                fmt(row.get("loss")),
                row.get("utterances"),
            ]
            for name, row in sorted(splits.items())
        ],
    )
    print_kv("Eval Metadata", [
        ("checkpoint", eval_results.get("checkpoint")),
        ("evaluated_at", eval_results.get("evaluated_at")),
        ("hostname", eval_results.get("hostname")),
    ])
else:
    print("\n== Test Results ==\nmissing eval_results.json")
PY
