"""Train one of {VQEMA, RVQ, ParallelVQ} on the PCA-64 chunks that the
k-means baseline uses, then save the trained quantizer for later target
extraction.

Pipeline:

1. Load CMVN + PCA + chunk_size/stride from the source targets directory's
   ``cluster_artifacts.joblib`` — these define the exact chunk space the
   k-means baseline operates in. Only the assignment mechanism varies.

2. Build (or reuse) a cached numpy file holding every PCA-64 chunk across
   train-960. The cache is ~22 GB on disk and is built once, then shared
   across all three quantizer training runs.

3. K-means-init the quantizer codebook on a 1M-chunk sample so it starts
   where the k-means baseline lands — isolates "frozen k-means vs
   EMA-learned" as the only axis.

4. Train: random batches → forward → EMA codebook update. Commitment
   loss is logged but never backpropped (no upstream encoder to update).
   Dead-code revival every ``--revival-interval`` steps.

5. Save state.pt with codebook + EMA accumulators + hyperparams, plus a
   metrics.json with final perplexity / active codes / dead-code rate.

Usage:

    python -m CausalSpecUnit.train_quantizer \\
        --source-targets-dir outputs/causal_specunit/targets_960h_c8 \\
        --data-root dataset/datasets/librispeech/LibriSpeech \\
        --output-dir outputs/causal_specunit/vq/rvq_100_500 \\
        --quantizer-type rvq --K1 100 --K2 500 \\
        --steps 30000 --batch-size 8192

Submit via slurm/causal_specunit/20_train_quantizer.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import joblib
import numpy as np
import torch
from tqdm import tqdm

from CausalSpecUnit.common import TRAIN_SPLITS
from CausalSpecUnit.data import LogMelExtractor, apply_cmvn, iter_librispeech_items, load_cmvn
from CausalSpecUnit.generate_targets import chunks_from_mel
from CausalSpecUnit.quantizer import RVQ, VQEMA, ParallelVQ


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Inputs
    p.add_argument("--source-targets-dir", required=True,
                   help="Existing k-means targets dir. Source for PCA, CMVN, chunk size/stride.")
    p.add_argument("--data-root", required=True,
                   help="LibriSpeech root containing train-clean-100 etc.")
    p.add_argument("--output-dir", required=True,
                   help="Where to save the trained quantizer (state.pt + metrics.json + train.jsonl).")
    p.add_argument("--splits", nargs="+", default=TRAIN_SPLITS,
                   help="LibriSpeech splits to extract chunks from.")
    p.add_argument("--chunks-cache", default=None,
                   help="Path to the PCA-64 chunks .npy cache. Default: "
                        "{source-targets-dir}/pca64_chunks.npy. Built if absent.")
    # Quantizer
    p.add_argument("--quantizer-type", choices=["vq", "rvq", "parallel"], required=True)
    p.add_argument("--K1", type=int, required=True,
                   help="Codebook size for VQ; level-1 size for RVQ/parallel.")
    p.add_argument("--K2", type=int, default=None,
                   help="Level-2 codebook size for RVQ/parallel. Required if --quantizer-type != vq.")
    p.add_argument("--beta", type=float, default=0.25,
                   help="Commitment-loss weight (logged only).")
    p.add_argument("--decay", type=float, default=0.99,
                   help="EMA decay rate for codebook updates.")
    # Training
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--kmeans-init-samples", type=int, default=1_000_000)
    p.add_argument("--revival-interval", type=int, default=1000)
    p.add_argument("--revival-threshold", type=float, default=0.001)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    # Misc
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Chunks cache: build once, reuse across quantizer types
# ---------------------------------------------------------------------------

def load_source_artifacts(source_dir: str):
    """Load PCA + chunk geometry from the existing k-means targets dir."""
    artifacts_path = os.path.join(source_dir, "cluster_artifacts.joblib")
    cmvn_path = os.path.join(source_dir, "cmvn.pt")
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Missing {artifacts_path} — run generate_targets.py first.")
    if not os.path.exists(cmvn_path):
        raise FileNotFoundError(f"Missing {cmvn_path}.")
    artifacts = joblib.load(artifacts_path)
    mean, std = load_cmvn(cmvn_path)
    return {
        "pca": artifacts.get("pca"),
        "mean": mean,
        "std": std,
        "chunk_size": artifacts["chunk_size"],
        "chunk_stride": artifacts["chunk_stride"],
        "pca_dim": artifacts.get("pca_dim", 64),
    }


def build_chunks_cache(
    data_root: str,
    splits: list,
    artifacts: dict,
    output_path: str,
) -> str:
    """Extract every PCA-64 chunk in train-960 and save as a single .npy."""
    mel_extractor = LogMelExtractor()
    pca = artifacts["pca"]
    mean = artifacts["mean"]
    std = artifacts["std"]
    chunk_size = artifacts["chunk_size"]
    chunk_stride = artifacts["chunk_stride"]

    items = list(iter_librispeech_items(data_root, splits))
    print(f"[cache] Extracting chunks from {len(items)} utterances across {splits} ...", flush=True)

    buffers = []
    total_chunks = 0
    for item in tqdm(items, desc="extract chunks"):
        mel = mel_extractor(item["audio_path"])
        mel = apply_cmvn(mel, mean, std)
        chunks = chunks_from_mel(mel, chunk_size, chunk_stride)
        if chunks.numel() == 0:
            continue
        chunks_np = chunks.numpy().astype(np.float32)
        if pca is not None:
            chunks_np = pca.transform(chunks_np)
        buffers.append(chunks_np)
        total_chunks += chunks_np.shape[0]

    all_chunks = np.concatenate(buffers, axis=0)
    del buffers
    print(f"[cache] Total chunks: {all_chunks.shape}, dtype={all_chunks.dtype}", flush=True)
    print(f"[cache] Saving to {output_path} (~{all_chunks.nbytes / 1e9:.1f} GB)...", flush=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, all_chunks)
    return output_path


def load_chunks_cache(path: str) -> np.ndarray:
    """Memory-map the chunks cache so we don't OOM on the full 22 GB."""
    return np.load(path, mmap_mode="r")


# ---------------------------------------------------------------------------
# Quantizer factory
# ---------------------------------------------------------------------------

def build_quantizer(args: argparse.Namespace, dim: int):
    if args.quantizer_type == "vq":
        return VQEMA(dim=dim, K=args.K1, beta=args.beta, decay=args.decay)
    if args.K2 is None:
        raise ValueError("--K2 is required for quantizer-type in {rvq, parallel}")
    cls = RVQ if args.quantizer_type == "rvq" else ParallelVQ
    return cls(dim=dim, K1=args.K1, K2=args.K2, beta=args.beta, decay=args.decay)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def is_two_stream(quantizer) -> bool:
    return isinstance(quantizer, (RVQ, ParallelVQ))


def step_forward(quantizer, x: torch.Tensor):
    """Returns (commit_loss, ppl_data) — ppl_data is (ppl,) for VQEMA and
    (ppl1, ppl2) for RVQ/ParallelVQ. Also performs the EMA update."""
    if is_two_stream(quantizer):
        _, indices_pair, commit, ppl_pair = quantizer(x)
        quantizer.update_codebooks(x, indices_pair)
        return commit, ppl_pair
    _, indices, commit, ppl = quantizer(x)
    quantizer.update_codebook(x, indices)
    return commit, (ppl,)


def active_codes_summary(quantizer, threshold: float) -> dict:
    if isinstance(quantizer, VQEMA):
        return {"active_q1": quantizer.active_codes(threshold), "K1": quantizer.K}
    return {
        "active_q1": quantizer.q1.active_codes(threshold),
        "K1": quantizer.q1.K,
        "active_q2": quantizer.q2.active_codes(threshold),
        "K2": quantizer.q2.K,
    }


def revive(quantizer, x: torch.Tensor, threshold: float):
    if isinstance(quantizer, VQEMA):
        return quantizer.revive_dead_codes(x, threshold)
    return quantizer.revive_dead_codes(x, threshold)


def train(args: argparse.Namespace, quantizer, chunks: np.ndarray, device: torch.device):
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "train.jsonl")
    open(metrics_path, "w").close()  # truncate

    def log(payload: dict) -> None:
        with open(metrics_path, "a") as f:
            f.write(json.dumps(payload) + "\n")

    log({"event": "train_start", "args": vars(args), "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
         "n_chunks": int(chunks.shape[0]), "dim": int(chunks.shape[1])})

    # ---------- k-means init ----------
    rng = np.random.default_rng(args.seed)
    n_init = min(args.kmeans_init_samples, chunks.shape[0])
    init_idx = rng.choice(chunks.shape[0], size=n_init, replace=False)
    init_sample = torch.from_numpy(np.array(chunks[init_idx]))
    print(f"[train] K-means init on {n_init} chunks...", flush=True)
    t0 = time.time()
    quantizer.init_from_kmeans(init_sample)
    quantizer.to(device)
    print(f"[train] K-means init done in {time.time() - t0:.1f}s", flush=True)

    # Post-init metrics on a held-out batch
    with torch.no_grad():
        eval_idx = rng.choice(chunks.shape[0], size=min(args.batch_size, chunks.shape[0]), replace=False)
        eval_batch = torch.from_numpy(np.array(chunks[eval_idx])).to(device)
        _, ppl_data = step_forward(quantizer, eval_batch)
    log({
        "event": "post_kmeans_init",
        **{f"ppl{i+1}": float(p.item()) for i, p in enumerate(ppl_data)},
        **active_codes_summary(quantizer, args.revival_threshold),
    })
    print(f"[train] post-init perplexity: {[f'{p.item():.1f}' for p in ppl_data]}", flush=True)

    # ---------- training loop ----------
    print(f"[train] training for {args.steps} steps, batch={args.batch_size}, decay={args.decay}", flush=True)
    N = chunks.shape[0]
    t_start = time.time()
    for step in range(1, args.steps + 1):
        batch_idx = rng.integers(0, N, size=args.batch_size)
        x = torch.from_numpy(np.array(chunks[batch_idx])).to(device)
        commit, ppl_data = step_forward(quantizer, x)

        if step % args.log_every == 0 or step == 1:
            log({
                "event": "step",
                "step": step,
                "commit": float(commit.item()),
                **{f"ppl{i+1}": float(p.item()) for i, p in enumerate(ppl_data)},
                **active_codes_summary(quantizer, args.revival_threshold),
                "elapsed_s": time.time() - t_start,
            })

        if step % args.revival_interval == 0:
            n_rev = revive(quantizer, x, args.revival_threshold)
            n_rev_total = n_rev if isinstance(n_rev, int) else sum(n_rev)
            if n_rev_total > 0:
                log({"event": "revive", "step": step, "n_revived": n_rev,
                     "elapsed_s": time.time() - t_start})

    # ---------- save ----------
    state_path = os.path.join(args.output_dir, "state.pt")
    torch.save({
        "quantizer_type": args.quantizer_type,
        "K1": args.K1,
        "K2": args.K2,
        "dim": chunks.shape[1],
        "beta": args.beta,
        "decay": args.decay,
        "state_dict": quantizer.state_dict(),
        "args": vars(args),
    }, state_path)
    print(f"[train] saved {state_path}", flush=True)

    # Final metrics: hold-out sample of the same size as a training batch
    with torch.no_grad():
        eval_idx = rng.choice(chunks.shape[0], size=min(args.batch_size, chunks.shape[0]), replace=False)
        eval_batch = torch.from_numpy(np.array(chunks[eval_idx])).to(device)
        _, ppl_data_final = step_forward(quantizer, eval_batch)
    summary = {
        "event": "train_end",
        "total_steps": args.steps,
        "elapsed_s": time.time() - t_start,
        **{f"ppl{i+1}": float(p.item()) for i, p in enumerate(ppl_data_final)},
        **active_codes_summary(quantizer, args.revival_threshold),
    }
    log(summary)

    metrics_summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[train] final perplexity: {[f'{p.item():.1f}' for p in ppl_data_final]}", flush=True)
    print(f"[train] done. wrote {state_path} and {metrics_summary_path}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    artifacts = load_source_artifacts(args.source_targets_dir)
    print(f"[main] source PCA dim={artifacts['pca_dim']} chunk={artifacts['chunk_size']}/"
          f"{artifacts['chunk_stride']}", flush=True)

    cache_path = args.chunks_cache or os.path.join(args.source_targets_dir, "pca64_chunks.npy")
    if not os.path.exists(cache_path):
        print(f"[main] No cache at {cache_path}; building...", flush=True)
        build_chunks_cache(args.data_root, args.splits, artifacts, cache_path)
    else:
        print(f"[main] Reusing chunks cache: {cache_path}", flush=True)
    chunks = load_chunks_cache(cache_path)
    print(f"[main] chunks shape={chunks.shape} dtype={chunks.dtype}", flush=True)

    device = torch.device(args.device)
    quantizer = build_quantizer(args, dim=chunks.shape[1])
    print(f"[main] built {args.quantizer_type} quantizer: "
          f"K1={args.K1}" + (f" K2={args.K2}" if args.K2 else ""), flush=True)

    train(args, quantizer, chunks, device)


if __name__ == "__main__":
    main()
