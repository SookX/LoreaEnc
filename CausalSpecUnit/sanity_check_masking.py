"""Sanity-check the SSL masking pipeline end-to-end.

Renders a few real LibriSpeech utterances through the same code path the
pretraining loop uses, then reports per-frame masked fraction, draws the
spectrogram and mask side-by-side, and prints summary statistics.

Run from the project root:
    python -m CausalSpecUnit.sanity_check_masking \
        --data-root dataset \
        --splits dev-clean \
        --targets-dir outputs/causal_specunit/targets_debug_c2 \
        --output-dir outputs/sanity_masking \
        --num-samples 6
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from CausalSpecUnit.data import SpecUnitDataset, collate_ssl
from CausalSpecUnit.model import CausalSpecUnitSSL
from CausalSpecUnit.pretrain_ssl import (
    corrupt_mel_from_target_mask,
    make_hubert_mask,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=["dev-clean"])
    p.add_argument("--targets-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs/sanity_masking")
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--mask-prob", type=float, default=0.08)
    p.add_argument("--mask-length", type=int, default=10)
    p.add_argument("--chunk-size", type=int, default=2)
    p.add_argument("--chunk-stride", type=int, default=4)
    p.add_argument("--variant", type=str, default="xs")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset ({args.splits}) ...")
    dataset = SpecUnitDataset(
        data_root=args.data_root,
        splits=args.splits,
        targets_path=os.path.join(args.targets_dir, "targets.pt"),
        cmvn_path=os.path.join(args.targets_dir, "cmvn.pt"),
        max_items=args.num_samples,
    )
    print(f"Dataset has {len(dataset)} items")

    # Build a single mini-batch by hand to mirror the training pipeline.
    items = [dataset[i] for i in range(min(args.num_samples, len(dataset)))]
    batch = collate_ssl(items)
    mel, lengths, z100, z500, target_lengths = batch
    mel = mel.to(device)

    print(f"\nBatch: mel={tuple(mel.shape)} dtype={mel.dtype} "
          f"min={mel.min().item():.2f} max={mel.max().item():.2f} "
          f"mean={mel.mean().item():.3f} std={mel.std().item():.3f}")

    # Build model just to get the learnable mask token.
    model = CausalSpecUnitSSL(variant=args.variant).to(device)
    mask_token = getattr(model, "mask_emb", None)
    has_token = isinstance(mask_token, torch.Tensor)
    print(f"Mask token: {'learnable nn.Parameter' if has_token else 'none (scalar)'}")

    # Generate the time mask in target space, then expand to mel space.
    masked_positions = make_hubert_mask(
        target_lengths=target_lengths,
        max_targets=z100.size(1),
        mask_prob=args.mask_prob,
        mask_length=args.mask_length,
        device=device,
    )
    corrupted = corrupt_mel_from_target_mask(
        mel=mel,
        masked_positions=masked_positions,
        chunk_size=args.chunk_size,
        chunk_stride=args.chunk_stride,
        mask_value=mask_token if has_token else 0.0,
    )

    target_frac = masked_positions.float().mean().item()
    # Reconstruct the per-mel-frame mask by diffing original and corrupted.
    diff = (mel != corrupted).any(dim=-1)  # (B, T)
    frame_frac = diff.float().mean().item()

    print(f"\nMasking stats (config: prob={args.mask_prob}, len={args.mask_length}, "
          f"c={args.chunk_size}, s={args.chunk_stride}):")
    print(f"  target-space masked fraction : {target_frac:.2%}")
    print(f"  mel-frame masked fraction    : {frame_frac:.2%}")
    print(f"  expected if no leakage       : ~{target_frac:.2%} "
          f"(should match target frac when stride covers whole window)")

    # Per-utterance breakdown.
    print(f"\nPer-utterance:")
    print(f"  {'idx':>3}  {'mel_T':>6}  {'tgt_T':>6}  {'tgt_msk%':>8}  {'frm_msk%':>8}")
    for i in range(mel.size(0)):
        T = int(lengths[i].item()) if lengths[i].item() <= mel.size(1) else mel.size(1)
        tT = int(target_lengths[i].item())
        tmf = masked_positions[i, :tT].float().mean().item()
        fmf = diff[i, :T].float().mean().item()
        print(f"  {i:>3}  {T:>6}  {tT:>6}  {tmf:>7.2%}   {fmf:>7.2%}")

    # Plot the first few utterances.
    n_plot = min(3, mel.size(0))
    fig, axes = plt.subplots(n_plot, 2, figsize=(12, 3 * n_plot))
    if n_plot == 1:
        axes = axes.reshape(1, 2)

    for i in range(n_plot):
        T = min(int(lengths[i].item()), mel.size(1))
        orig = mel[i, :T].cpu().numpy().T  # (n_mels, T)
        corr = corrupted[i, :T].cpu().numpy().T
        vmin, vmax = orig.min(), orig.max()

        axes[i, 0].imshow(orig, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
        axes[i, 0].set_title(f"Utt {i}: original mel ({T} frames)")
        axes[i, 0].set_xlabel("time frame"); axes[i, 0].set_ylabel("mel bin")

        axes[i, 1].imshow(corr, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
        axes[i, 1].set_title(f"Utt {i}: corrupted ({diff[i, :T].float().mean().item():.0%} masked)")
        axes[i, 1].set_xlabel("time frame"); axes[i, 1].set_ylabel("mel bin")

    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "masking_visualization.png")
    fig.savefig(plot_path, dpi=110)
    print(f"\nSaved visualization: {plot_path}")

    # Sanity: the relationship target_frac vs frame_frac should be approximately
    # target_frac (since stride covers each target's window exactly once).
    if abs(target_frac - frame_frac) > 0.05:
        print(f"\n[WARN] Large mismatch between target-space and mel-frame masking "
              f"({target_frac:.2%} vs {frame_frac:.2%}). Could indicate mask leakage or overshoot.")
    else:
        print(f"\n[OK] Target-space and mel-frame masking match within tolerance.")


if __name__ == "__main__":
    main()