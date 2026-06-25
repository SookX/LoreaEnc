"""Generate iter-2 VQ targets.

Same recipe as ``generate_iter2_targets.py`` but with learned VQ
(VQEMA / RVQ / ParallelVQ) instead of k-means as the quantizer fit on the
iter-1 encoder's hidden states.

Pipeline:
    1. Load the iter-1 SSL encoder (from --ssl-checkpoint).
    2. Iterate all train-960 utterances; collect a sample of encoder
       hidden states into a fit buffer.
    3. PCA-reduce the sample, then train a VQ on it (k-means init + EMA).
    4. Run the encoder again over every utterance, quantize each frame's
       hidden state, save (z100, z500) indices in the standard schema.
    5. Shard targets for the SSL pretrain dataloader.

Saves the same artefacts pretrain_ssl.py expects:
    targets.pt, metadata.json, cmvn.pt, cluster_artifacts.joblib,
    target_index.json + targets_shards/

Plus a quantizer_state.pt for debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from CausalSpecUnit.common import TRAIN_SPLITS
from CausalSpecUnit.generate_iter2_targets import (
    EncoderFeatureDataset,
    collect_fit_features,
    encode_batch,
    load_ssl_model,
    make_loader,
    write_target_shards,
)
from CausalSpecUnit.quantizer import RVQ, VQEMA, ParallelVQ


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Inputs
    p.add_argument("--data-root", required=True)
    p.add_argument("--splits", nargs="+", default=TRAIN_SPLITS)
    p.add_argument("--cmvn-path", required=True)
    p.add_argument("--ssl-checkpoint", required=True,
                   help="Iter-1 SSL checkpoint (encoder source for hidden-state extraction).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--variant", default="xs")
    # Geometry (carried into metadata so pretrain_ssl validates correctly)
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--chunk-stride", type=int, default=4)
    p.add_argument("--pca-dim", type=int, default=64,
                   help="PCA output dim on the encoder hidden states. 0 = skip PCA.")
    # Quantizer
    p.add_argument("--quantizer-type", choices=["vq", "rvq", "parallel"], required=True)
    p.add_argument("--K1", type=int, required=True)
    p.add_argument("--K2", type=int, default=None)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--decay", type=float, default=0.99)
    p.add_argument("--vq-steps", type=int, default=30000)
    p.add_argument("--vq-batch-size", type=int, default=8192)
    # Fit + assignment
    p.add_argument("--max-fit-frames", type=int, default=1_000_000)
    p.add_argument("--fit-frames-per-batch", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Per-utterance batch size for encoder forward.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dataloader-timeout", type=int, default=180)
    p.add_argument("--max-utterances", type=int, default=None,
                   help="Limit utterances for debugging (None = all).")
    p.add_argument("--target-shards", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_quantizer(args: argparse.Namespace, dim: int):
    if args.quantizer_type == "vq":
        return VQEMA(dim=dim, K=args.K1, beta=args.beta, decay=args.decay)
    if args.K2 is None:
        raise ValueError("--K2 required for rvq / parallel")
    cls = RVQ if args.quantizer_type == "rvq" else ParallelVQ
    return cls(dim=dim, K1=args.K1, K2=args.K2, beta=args.beta, decay=args.decay)


def fit_vq(features: np.ndarray, args: argparse.Namespace, device: torch.device):
    """PCA-reduce + train a VQ on the encoder fit buffer. Returns (pca, quantizer)."""
    if args.pca_dim > 0:
        print(f"Fitting PCA ({features.shape[1]} -> {args.pca_dim}) on {features.shape[0]:,} encoder frames")
        pca = PCA(n_components=args.pca_dim, whiten=True, random_state=args.seed)
        reduced = pca.fit_transform(features)
    else:
        print(f"Skipping PCA; quantizing {features.shape[1]}-dim encoder frames directly")
        pca = None
        reduced = features
    reduced = np.ascontiguousarray(reduced, dtype=np.float32)

    print(f"Building {args.quantizer_type} quantizer: K1={args.K1}" +
          (f" K2={args.K2}" if args.K2 else ""))
    quantizer = build_quantizer(args, dim=reduced.shape[1])

    print(f"K-means init on {reduced.shape[0]:,} reduced frames")
    quantizer.init_from_kmeans(torch.from_numpy(reduced))
    quantizer.to(device)

    print(f"EMA training {args.vq_steps} steps, batch {args.vq_batch_size}")
    is_two_stream = args.quantizer_type != "vq"
    rng = np.random.default_rng(args.seed)
    n = reduced.shape[0]
    t0 = time.time()
    for step in range(1, args.vq_steps + 1):
        idx = rng.integers(0, n, size=args.vq_batch_size)
        x = torch.from_numpy(reduced[idx]).to(device)
        if is_two_stream:
            _, indices_pair, _, _ = quantizer(x)
            quantizer.update_codebooks(x, indices_pair)
        else:
            _, indices, _, _ = quantizer(x)
            quantizer.update_codebook(x, indices)
        if step % 5000 == 0:
            print(f"  step {step}/{args.vq_steps}  elapsed={time.time() - t0:.0f}s", flush=True)

    print(f"VQ training done in {time.time() - t0:.0f}s")
    return pca, quantizer


@torch.inference_mode()
def assign_targets_vq(model, pca, quantizer, args: argparse.Namespace, device: torch.device):
    """Run encoder over every utterance, quantize, return {uid: {z100, z500}}."""
    dataset = EncoderFeatureDataset(
        args.data_root, args.splits, args.cmvn_path,
        max_utterances=args.max_utterances, shuffle_items=False, seed=args.seed,
    )
    loader = make_loader(dataset, args)

    is_two_stream = args.quantizer_type != "vq"
    targets: dict = {}
    for uids, _, mels, lengths in tqdm(loader, desc="assign iter-2 VQ"):
        encoded, out_lengths = encode_batch(model, mels, lengths, device)
        for i in range(encoded.size(0)):
            valid = int(out_lengths[i].item())
            if valid <= 0:
                continue
            feat = encoded[i, :valid].numpy().astype(np.float32, copy=False)
            if pca is not None:
                feat = pca.transform(feat).astype(np.float32, copy=False)
            x = torch.from_numpy(feat).to(device)
            if is_two_stream:
                _, (i1, i2), _, _ = quantizer(x)
                targets[uids[i]] = {
                    "z100": i1.cpu().long(),
                    "z500": i2.cpu().long(),
                }
            else:
                _, indices, _, _ = quantizer(x)
                z = indices.cpu().long()
                targets[uids[i]] = {
                    "z100": torch.zeros_like(z),
                    "z500": z,
                }
    return targets


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: load iter-1 SSL encoder
    print(f"Loading iter-1 SSL encoder from {args.ssl_checkpoint}")
    model, missing, unexpected, _ = load_ssl_model(args.ssl_checkpoint, args.variant, device)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")

    # Step 2: collect a sample of encoder frames for fitting
    features, n_dataset, seen_frames = collect_fit_features(model, args, device)
    print(f"Collected {features.shape[0]:,} fit frames "
          f"({seen_frames:,} seen across {n_dataset} utterances)")

    # Step 3: PCA + VQ fit
    pca, quantizer = fit_vq(features, args, device)

    # Save quantizer + PCA so we can audit and resume
    qstate_path = os.path.join(args.output_dir, "quantizer_state.pt")
    torch.save({
        "quantizer_type": args.quantizer_type,
        "K1": args.K1,
        "K2": args.K2,
        "dim": args.pca_dim if args.pca_dim > 0 else features.shape[1],
        "beta": args.beta,
        "decay": args.decay,
        "state_dict": quantizer.state_dict(),
        "args": vars(args),
    }, qstate_path)
    joblib.dump(
        {
            "pca": pca,
            "k_coarse": args.K1 if args.quantizer_type != "vq" else 100,
            "k_fine": args.K2 if args.K2 is not None else args.K1,
            "chunk_size": args.chunk_size,
            "chunk_stride": args.chunk_stride,
            "pca_dim": args.pca_dim,
            "num_fit_frames": features.shape[0],
            "quantizer_type": args.quantizer_type,
        },
        os.path.join(args.output_dir, "cluster_artifacts.joblib"),
    )
    print(f"Saved quantizer state to {qstate_path}")
    del features

    # Step 4: assign per-utterance targets
    print(f"Assigning iter-2 VQ targets")
    targets = assign_targets_vq(model, pca, quantizer, args, device)

    # Step 5: save
    targets_path = os.path.join(args.output_dir, "targets.pt")
    torch.save(targets, targets_path)
    print(f"Wrote {len(targets)} utterances to {targets_path}")

    shutil.copy(args.cmvn_path, os.path.join(args.output_dir, "cmvn.pt"))

    if args.quantizer_type == "vq":
        k_coarse, k_fine = 100, args.K1
    else:
        k_coarse, k_fine = args.K1, args.K2
    metadata = {
        "chunk_size": args.chunk_size,
        "chunk_stride": args.chunk_stride,
        "k_coarse": k_coarse,
        "k_fine": k_fine,
        "pca_dim": args.pca_dim,
        "quantizer_type": args.quantizer_type,
        "is_two_stream": args.quantizer_type != "vq",
        "num_target_utterances": len(targets),
        "target_features": "ssl_encoder_iter2_vq",
        "source_ssl_checkpoint": args.ssl_checkpoint,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    write_target_shards(targets, args.output_dir, args.target_shards)
    print("Done.")


if __name__ == "__main__":
    main()
