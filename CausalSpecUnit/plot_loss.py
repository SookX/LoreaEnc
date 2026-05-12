"""
Plot SSL pretraining loss curve from saved checkpoints.

Usage:
    python -m CausalSpecUnit.plot_loss --pretrain-dir outputs/causal_specunit/pretrain
    python -m CausalSpecUnit.plot_loss --pretrain-dir outputs/causal_specunit/pretrain --output loss.png
"""

import argparse
import os
import sys

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain-dir", type=str, required=True,
                   help="Output directory from pretrain_ssl.py containing checkpoint_stepXXXXXX dirs.")
    p.add_argument("--output", type=str, default=None,
                   help="Save plot to this path. If omitted, saves next to pretrain-dir as loss_curve.png.")
    return p.parse_args()


def load_checkpoints(pretrain_dir):
    entries = []
    for name in os.listdir(pretrain_dir):
        if not name.startswith("checkpoint_step"):
            continue
        try:
            step = int(name.replace("checkpoint_step", ""))
        except ValueError:
            continue
        ckpt_path = os.path.join(pretrain_dir, name, "checkpoint.pt")
        if not os.path.isfile(ckpt_path):
            continue
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ssl_loss = state.get("ssl_loss")
        epoch = state.get("epoch")
        if ssl_loss is not None:
            entries.append((step, epoch, float(ssl_loss)))
    entries.sort()
    return entries


def main():
    args = parse_args()

    if not os.path.isdir(args.pretrain_dir):
        print(f"Directory not found: {args.pretrain_dir}")
        sys.exit(1)

    entries = load_checkpoints(args.pretrain_dir)
    if not entries:
        print("No checkpoints with ssl_loss found. Has pretraining started yet?")
        sys.exit(1)

    steps = [e[0] for e in entries]
    losses = [e[2] for e in entries]

    print(f"Loaded {len(entries)} checkpoints")
    print(f"  Steps  : {steps[0]} → {steps[-1]}")
    print(f"  Loss   : {losses[0]:.4f} → {losses[-1]:.4f}")
    drop = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"  Drop   : {drop:.1f}%")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import math

        random_baseline_k100 = math.log(100)
        random_baseline_k500 = math.log(500)
        combined_baseline = random_baseline_k100 + random_baseline_k500

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, losses, marker="o", markersize=4, color="steelblue", label="SSL loss (k100+k500)")
        ax.axhline(combined_baseline, color="red", linestyle="--", alpha=0.6,
                   label=f"random baseline ({combined_baseline:.2f})")
        ax.axhline(random_baseline_k100, color="orange", linestyle=":", alpha=0.6,
                   label=f"random baseline k=100 ({random_baseline_k100:.2f})")

        # Shade good zone
        ax.axhspan(0, combined_baseline * 0.8, alpha=0.05, color="green")

        ax.set_xlabel("Optimizer steps")
        ax.set_ylabel("SSL Loss")
        ax.set_title("CausalSpecUnit SSL Pretraining Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out = args.output or os.path.join(args.pretrain_dir, "loss_curve.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"\nPlot saved to: {out}")

    except ImportError:
        print("\nmatplotlib not installed — install with: pip install matplotlib")
        print("Raw values:")
        for step, epoch, loss in entries:
            print(f"  step={step:6d}  epoch={epoch:3d}  loss={loss:.4f}")

    # Go/no-go verdict
    print()
    if len(entries) < 3:
        print("Verdict: WAIT — not enough checkpoints yet to judge trend")
    elif drop < 2.0:
        print("Verdict: NO-GO — loss is not decreasing (<2% drop). Check masking, LR, or targets.")
    elif losses[-1] > combined_baseline:
        print(f"Verdict: CAUTION — loss is decreasing ({drop:.1f}%) but still above random baseline.")
        print("         May need more steps before it becomes useful.")
    else:
        print(f"Verdict: GO — loss dropped {drop:.1f}% and is below random baseline.")


if __name__ == "__main__":
    main()
