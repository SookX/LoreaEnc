#!/usr/bin/env python3
"""Summarize SSL pretraining cost from CausalSpecUnit JSONL logs.

The goal is to report quantities that are meaningful when comparing against
large SSL systems: encoder size, pretraining data, optimizer steps, wall time,
GPU-hours, and throughput. GPU-hours are intentionally reported without
hardware normalization; include the GPU model separately in the paper text.
"""

import argparse
import json
import math
from pathlib import Path


LITERATURE_PRESETS = {
    "wav2vec2-base-ls960": {
        "system": "wav2vec 2.0 Base",
        "encoder_params": 95_000_000,
        "pretrain_data_hours": 960.0,
        "world_size": 64,
        "wall_hours": 1.6 * 24.0,
        "gpu_hours": 64 * 1.6 * 24.0,
        "optimizer_steps": 400_000,
        "gpu_name": "V100",
        "iterations": 1,
        "source": "Baevski et al. 2020: 64 V100 GPUs for 1.6 days; 400k updates.",
    },
    "wav2vec2-large-ls960": {
        "system": "wav2vec 2.0 Large",
        "encoder_params": 317_000_000,
        "pretrain_data_hours": 960.0,
        "world_size": 128,
        "wall_hours": 2.3 * 24.0,
        "gpu_hours": 128 * 2.3 * 24.0,
        "optimizer_steps": 250_000,
        "gpu_name": "V100",
        "iterations": 1,
        "source": "Baevski et al. 2020: 128 V100 GPUs for 2.3 days on LibriSpeech; 250k updates.",
    },
    "hubert-base-ls960": {
        "system": "HuBERT Base iter1+iter2",
        "encoder_params": 95_000_000,
        "pretrain_data_hours": 960.0,
        "world_size": 32,
        "wall_hours": 9.5 * 6.5,
        "gpu_hours": 32 * 9.5 * 6.5,
        "optimizer_steps": 650_000,
        "gpu_name": "not specified",
        "iterations": 2,
        "source": "Hsu et al. 2021: Base uses 32 GPUs; iter1=250k and iter2=400k; 100k steps take about 9.5h.",
    },
    "hubert-large-ll60k": {
        "system": "HuBERT Large",
        "encoder_params": 317_000_000,
        "pretrain_data_hours": 60_000.0,
        "world_size": 128,
        "wall_hours": None,
        "gpu_hours": None,
        "optimizer_steps": 400_000,
        "gpu_name": "not specified",
        "iterations": 1,
        "source": "Hsu et al. 2021: Large uses 128 GPUs for 400k steps on Libri-Light; wall time not reported.",
    },
}


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def human_params(value):
    if value is None:
        return "?"
    value = float(value)
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return str(int(value))


def fmt_float(value, digits=2):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "?"
    return f"{value:.{digits}f}"


def latest_record(rows):
    candidates = [
        row for row in rows
        if row.get("event") in {"train_step", "epoch_end"} and "elapsed_hours" in row
    ]
    if not candidates:
        raise ValueError("No train_step/epoch_end records with elapsed_hours found")
    return max(candidates, key=lambda row: float(row.get("optimizer_step", -1)))


def run_start(rows):
    for row in rows:
        if row.get("event") == "run_start":
            return row
    return {}


def load_run_info(metrics_path, explicit_path):
    if explicit_path:
        path = Path(explicit_path)
    else:
        path = Path(metrics_path).with_name("ssl_run_info.json")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize(args):
    rows = read_jsonl(args.ssl_metrics)
    start = run_start(rows)
    info = load_run_info(args.ssl_metrics, args.run_info)
    end = latest_record(rows)

    source = info or start
    train_args = source.get("args", {})
    metadata = source.get("metadata", {})
    params = source.get("parameter_counts", {})

    world_size = args.gpus or source.get("world_size") or start.get("world_size") or 1
    wall_hours = float(end["elapsed_hours"])
    gpu_hours = wall_hours * int(world_size)
    optimizer_steps = int(end.get("optimizer_step", 0))

    batch_size = train_args.get("batch_size")
    grad_accum = train_args.get("grad_accum_steps", 1)
    effective_batch = None
    if batch_size is not None:
        effective_batch = int(batch_size) * int(world_size) * int(grad_accum)

    utterances_seen = optimizer_steps * effective_batch if effective_batch else None
    steps_per_hour = optimizer_steps / wall_hours if wall_hours > 0 else None
    utterances_per_sec = (
        utterances_seen / (wall_hours * 3600.0)
        if utterances_seen is not None and wall_hours > 0
        else None
    )

    pretrain_hours = args.pretrain_hours
    if pretrain_hours is None:
        splits = metadata.get("splits") or []
        if set(splits) == {"train-clean-100", "train-clean-360", "train-other-500"}:
            pretrain_hours = 960.0

    epochs_equiv = None
    audio_hours_processed = None
    if utterances_seen is not None and metadata.get("num_target_utterances"):
        epochs_equiv = utterances_seen / float(metadata["num_target_utterances"])
        if pretrain_hours is not None:
            audio_hours_processed = epochs_equiv * pretrain_hours

    report = {
        "system": args.system,
        "ssl_metrics": str(args.ssl_metrics),
        "hostname": source.get("hostname"),
        "variant": train_args.get("variant"),
        "encoder_params": params.get("encoder"),
        "total_params": params.get("total"),
        "pretrain_data_hours": pretrain_hours,
        "target_features": metadata.get("target_features"),
        "target_utterances": metadata.get("num_target_utterances"),
        "world_size": int(world_size),
        "wall_hours": wall_hours,
        "gpu_hours": gpu_hours,
        "optimizer_steps": optimizer_steps,
        "effective_batch": effective_batch,
        "steps_per_hour": steps_per_hour,
        "utterances_per_sec": utterances_per_sec,
        "epochs_equiv": epochs_equiv,
        "audio_hours_processed": audio_hours_processed,
        "final_ssl_loss": end.get("loss"),
        "final_c100": end.get("c100"),
        "final_c500": end.get("c500"),
        "gpu_name": args.gpu_name,
        "iterations": 1,
    }
    return report


def combine_reports(reports, system):
    if len(reports) == 1:
        report = dict(reports[0])
        report["system"] = system
        return report

    combined = dict(reports[-1])
    combined["system"] = system
    combined["ssl_metrics"] = ", ".join(str(report["ssl_metrics"]) for report in reports)
    combined["wall_hours"] = sum(float(report["wall_hours"]) for report in reports)
    combined["gpu_hours"] = sum(float(report["gpu_hours"]) for report in reports)
    combined["optimizer_steps"] = sum(int(report["optimizer_steps"]) for report in reports)
    combined["iterations"] = sum(int(report.get("iterations") or 0) for report in reports)
    combined["final_ssl_loss"] = reports[-1].get("final_ssl_loss")
    combined["final_c100"] = reports[-1].get("final_c100")
    combined["final_c500"] = reports[-1].get("final_c500")
    if combined["wall_hours"] > 0:
        combined["steps_per_hour"] = combined["optimizer_steps"] / combined["wall_hours"]
    return combined


def manual_report(spec, base_report):
    parts = spec.split(":")
    if len(parts) not in {3, 5}:
        raise ValueError(
            "--manual-local format is label:steps:wall_hours[:gpus:iterations], "
            "for example iter2:100000:8.4:2:1"
        )
    label = parts[0]
    steps = int(parts[1])
    wall_hours = float(parts[2])
    gpus = int(parts[3]) if len(parts) == 5 else int(base_report.get("world_size") or 1)
    iterations = int(parts[4]) if len(parts) == 5 else 1

    report = dict(base_report)
    report["system"] = label
    report["ssl_metrics"] = "manual"
    report["optimizer_steps"] = steps
    report["wall_hours"] = wall_hours
    report["world_size"] = gpus
    report["gpu_hours"] = wall_hours * gpus
    report["iterations"] = iterations
    report["steps_per_hour"] = steps / wall_hours if wall_hours > 0 else None
    return report


def print_report(report):
    print("Pretraining cost report")
    print(f"  system:              {report['system']}")
    print(f"  metrics:             {report['ssl_metrics']}")
    if report.get("hostname"):
        print(f"  host:                {report['hostname']}")
    if report.get("gpu_name"):
        print(f"  gpu:                 {report['gpu_name']}")
    print(f"  variant:             {report.get('variant') or '?'}")
    print(f"  encoder params:      {human_params(report.get('encoder_params'))}")
    print(f"  total SSL params:    {human_params(report.get('total_params'))}")
    print(f"  pretrain data:       {fmt_float(report.get('pretrain_data_hours'), 0)} h")
    print(f"  target features:     {report.get('target_features') or '?'}")
    print(f"  target utterances:   {report.get('target_utterances') or '?'}")
    print(f"  GPUs:                {report['world_size']}")
    print(f"  wall time:           {fmt_float(report['wall_hours'], 2)} h")
    print(f"  GPU-hours:           {fmt_float(report['gpu_hours'], 2)}")
    print(f"  optimizer steps:     {report['optimizer_steps']:,}")
    print(f"  effective batch:     {report.get('effective_batch') or '?'} utterances")
    print(f"  throughput:          {fmt_float(report.get('steps_per_hour'), 1)} steps/h")
    print(f"  utterance throughput:{fmt_float(report.get('utterances_per_sec'), 1)} utt/s")
    print(f"  dataset passes:      {fmt_float(report.get('epochs_equiv'), 2)}")
    print(f"  audio-hours seen:    {fmt_float(report.get('audio_hours_processed'), 0)} h")
    print(f"  final SSL loss:      {fmt_float(report.get('final_ssl_loss'), 3)}")
    print(f"  iterations:          {report.get('iterations') or '?'}")


def latex_row(report):
    params = human_params(report.get("encoder_params"))
    data_hours = fmt_float(report.get("pretrain_data_hours"), 0)
    gpu_hours = fmt_float(report.get("gpu_hours"), 1)
    steps = f"{report['optimizer_steps'] // 1000}k" if report["optimizer_steps"] >= 1000 else str(report["optimizer_steps"])
    return (
        f"{report['system']} & {params} & {data_hours}h & "
        f"{report.get('iterations') or '?'} & {report.get('world_size') or '?'} & "
        f"{steps} & {gpu_hours} \\\\"
    )


def literature_report(name):
    if name not in LITERATURE_PRESETS:
        choices = ", ".join(sorted(LITERATURE_PRESETS))
        raise ValueError(f"Unknown literature preset {name!r}. Choices: {choices}")
    row = dict(LITERATURE_PRESETS[name])
    row.setdefault("ssl_metrics", "literature")
    row.setdefault("hostname", None)
    row.setdefault("variant", None)
    row.setdefault("total_params", None)
    row.setdefault("target_features", None)
    row.setdefault("target_utterances", None)
    row.setdefault("effective_batch", None)
    row.setdefault("steps_per_hour", None)
    row.setdefault("utterances_per_sec", None)
    row.setdefault("epochs_equiv", None)
    row.setdefault("audio_hours_processed", None)
    row.setdefault("final_ssl_loss", None)
    row.setdefault("final_c100", None)
    row.setdefault("final_c500", None)
    row.setdefault("iterations", None)
    return row


def print_comparison(reports):
    headers = ["System", "Params", "Data", "Iters", "GPUs", "Steps", "Wall h", "GPU-h"]
    rows = []
    for report in reports:
        steps = "?"
        if report.get("optimizer_steps") is not None:
            steps = (
                f"{report['optimizer_steps'] // 1000}k"
                if report["optimizer_steps"] >= 1000
                else str(report["optimizer_steps"])
            )
        rows.append([
            report["system"],
            human_params(report.get("encoder_params")),
            f"{fmt_float(report.get('pretrain_data_hours'), 0)}h",
            str(report.get("iterations") or "?"),
            str(report.get("world_size") or "?"),
            steps,
            fmt_float(report.get("wall_hours"), 1),
            fmt_float(report.get("gpu_hours"), 1),
        ])

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def line(values):
        return "  ".join(value.ljust(width) for value, width in zip(values, widths))

    print(line(headers))
    print(line(["-" * width for width in widths]))
    for row in rows:
        print(line(row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ssl_metrics", nargs="*", help="Path(s) to ssl_metrics.jsonl. Pass iter-1 and iter-2 logs to sum pretraining cost.")
    parser.add_argument("--run-info", default=None, help="Optional ssl_run_info.json path")
    parser.add_argument("--system", default="Ours (SF-XS)")
    parser.add_argument("--gpus", type=int, default=None, help="Override GPU count/world size")
    parser.add_argument("--gpu-name", default=None, help="GPU model, e.g. A100-80GB")
    parser.add_argument("--pretrain-hours", type=float, default=None,
                        help="Unlabeled pretraining corpus hours. Inferred as 960 for full LibriSpeech.")
    parser.add_argument("--literature", action="append", default=[],
                        choices=sorted(LITERATURE_PRESETS),
                        help="Add a literature preset to a comparison table. Can be passed multiple times.")
    parser.add_argument("--list-literature", action="store_true",
                        help="List available literature presets and exit.")
    parser.add_argument("--comparison", action="store_true",
                        help="Print a compact comparison table for ours plus requested literature presets.")
    parser.add_argument("--manual-local", action="append", default=[],
                        help="Add a local run without a metrics file as label:steps:wall_hours[:gpus:iterations]. "
                             "Useful when only the step count is known; use a measured or explicitly estimated wall time.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--latex-row", action="store_true", help="Print a LaTeX table row")
    args = parser.parse_args()

    if args.list_literature:
        for name, preset in LITERATURE_PRESETS.items():
            print(f"{name}: {preset['source']}")
        return

    reports = []
    local_reports = []
    if args.ssl_metrics:
        for metrics_path in args.ssl_metrics:
            one_args = argparse.Namespace(**vars(args))
            one_args.ssl_metrics = metrics_path
            local_reports.append(summarize(one_args))
    if args.manual_local:
        if local_reports:
            base_report = local_reports[-1]
        else:
            base_report = {
                "system": args.system,
                "ssl_metrics": "manual",
                "encoder_params": None,
                "total_params": None,
                "pretrain_data_hours": args.pretrain_hours,
                "world_size": args.gpus or 1,
                "gpu_name": args.gpu_name,
                "variant": None,
                "target_features": None,
                "target_utterances": None,
                "effective_batch": None,
                "utterances_per_sec": None,
                "epochs_equiv": None,
                "audio_hours_processed": None,
                "final_ssl_loss": None,
                "final_c100": None,
                "final_c500": None,
            }
        local_reports.extend(manual_report(spec, base_report) for spec in args.manual_local)
    if local_reports:
        reports.append(combine_reports(local_reports, args.system))
    reports.extend(literature_report(name) for name in args.literature)

    if not reports:
        raise SystemExit("Pass ssl_metrics or at least one --literature preset.")

    report = reports[0]
    if args.json:
        payload = reports if len(reports) > 1 else report
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.comparison:
        print_comparison(reports)
    else:
        print_report(report)
    if args.latex_row:
        print()
        print("LaTeX rows:")
        for item in reports:
            print(latex_row(item))


if __name__ == "__main__":
    main()
