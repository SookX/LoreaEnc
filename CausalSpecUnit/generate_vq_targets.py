"""Extract per-utterance VQ targets from a trained quantizer.

Reads:
  --source-targets-dir : existing k-means targets dir; we reuse its
                         PCA, CMVN, and chunk geometry so VQ cells share
                         the exact chunk space with k-means cells.
  --quantizer-dir      : output of train_quantizer.py; contains state.pt.

Writes:
  --output-dir/targets.pt          : {uid: {z100, z500}} for all utterances
  --output-dir/metadata.json       : k_coarse, k_fine, chunk geometry, etc.
  --output-dir/cmvn.pt             : copy of source CMVN
  --output-dir/cluster_artifacts.joblib : copy of source artifacts (PCA)

Schema:
  - RVQ / ParallelVQ: z100 = level-1/codebook-1 indices, z500 = level-2/codebook-2 indices.
    Drops in to the existing dual-codebook SSL pipeline with K_c=K1, K_f=K2.
  - Flat VQ-K: z500 = indices, z100 = zeros (dummy, never predicted because
    SSL is launched with --codebook-mode fine). Keeps the schema uniform so
    the existing data loader doesn't need to change.

Usage:
    python -m CausalSpecUnit.generate_vq_targets \\
        --source-targets-dir outputs/causal_specunit/targets_960h_c8 \\
        --quantizer-dir outputs/causal_specunit/vq/rvq_100_500 \\
        --output-dir outputs/causal_specunit/targets_960h_c8_rvq_100_500 \\
        --data-root dataset/datasets/librispeech/LibriSpeech
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
from tqdm import tqdm

from CausalSpecUnit.common import TRAIN_SPLITS
from CausalSpecUnit.data import LogMelExtractor, apply_cmvn, iter_librispeech_items, load_cmvn
from CausalSpecUnit.generate_targets import chunks_from_mel
from CausalSpecUnit.quantizer import RVQ, VQEMA, ParallelVQ


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-targets-dir", required=True,
                   help="Existing k-means targets dir to inherit PCA, CMVN, chunk geometry from.")
    p.add_argument("--quantizer-dir", required=True,
                   help="Output dir of train_quantizer.py; must contain state.pt.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-root", required=True,
                   help="LibriSpeech root.")
    p.add_argument("--splits", nargs="+", default=TRAIN_SPLITS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quantize-batch-size", type=int, default=8192,
                   help="Batch size for per-utterance chunk quantization on GPU.")
    return p.parse_args()


def load_quantizer(quantizer_dir: str):
    """Load a saved quantizer + return (module, type, K1, K2)."""
    state_path = os.path.join(quantizer_dir, "state.pt")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Missing {state_path}. Run train_quantizer.py first.")
    sd = torch.load(state_path, map_location="cpu", weights_only=False)
    qtype = sd["quantizer_type"]
    K1 = int(sd["K1"])
    K2 = int(sd["K2"]) if sd.get("K2") is not None else None
    dim = int(sd["dim"])
    beta = float(sd.get("beta", 0.25))
    decay = float(sd.get("decay", 0.99))

    if qtype == "vq":
        q = VQEMA(dim=dim, K=K1, beta=beta, decay=decay)
    elif qtype == "rvq":
        assert K2 is not None
        q = RVQ(dim=dim, K1=K1, K2=K2, beta=beta, decay=decay)
    elif qtype == "parallel":
        assert K2 is not None
        q = ParallelVQ(dim=dim, K1=K1, K2=K2, beta=beta, decay=decay)
    else:
        raise ValueError(f"Unknown quantizer_type: {qtype}")
    q.load_state_dict(sd["state_dict"])
    q.eval()
    return q, qtype, K1, K2


@torch.no_grad()
def quantize_chunks(q, chunks: torch.Tensor, device: torch.device, batch_size: int):
    """Quantize a (N, D) tensor of PCA-64 chunks.
    Returns one of:
      VQEMA  -> {'single': (N,) int64}
      RVQ / ParallelVQ -> {'i1': (N,) int64, 'i2': (N,) int64}
    """
    n = chunks.shape[0]
    if isinstance(q, VQEMA):
        idx_buf = []
        for i in range(0, n, batch_size):
            x = chunks[i:i + batch_size].to(device)
            _, idx, _, _ = q(x)
            idx_buf.append(idx.cpu())
        return {"single": torch.cat(idx_buf).long()}
    # Two-stream path (RVQ or ParallelVQ)
    i1_buf, i2_buf = [], []
    for i in range(0, n, batch_size):
        x = chunks[i:i + batch_size].to(device)
        _, (i1, i2), _, _ = q(x)
        i1_buf.append(i1.cpu())
        i2_buf.append(i2.cpu())
    return {"i1": torch.cat(i1_buf).long(), "i2": torch.cat(i2_buf).long()}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # ---------- Load source artifacts (PCA, CMVN, chunk geometry) ----------
    src_artifacts_path = os.path.join(args.source_targets_dir, "cluster_artifacts.joblib")
    src_cmvn_path = os.path.join(args.source_targets_dir, "cmvn.pt")
    if not os.path.isfile(src_artifacts_path):
        raise FileNotFoundError(f"Missing {src_artifacts_path}")
    if not os.path.isfile(src_cmvn_path):
        raise FileNotFoundError(f"Missing {src_cmvn_path}")
    artifacts = joblib.load(src_artifacts_path)
    pca = artifacts.get("pca")
    chunk_size = int(artifacts["chunk_size"])
    chunk_stride = int(artifacts["chunk_stride"])
    pca_dim = int(artifacts.get("pca_dim", 64))
    mean, std = load_cmvn(src_cmvn_path)
    print(f"[gen] source PCA dim={pca_dim} chunk={chunk_size}/{chunk_stride}", flush=True)

    # ---------- Load quantizer ----------
    q, qtype, K1, K2 = load_quantizer(args.quantizer_dir)
    q.to(device)
    is_two_stream = qtype != "vq"
    print(f"[gen] loaded {qtype} quantizer: K1={K1}" +
          (f" K2={K2}" if K2 else "") +
          f" two_stream={is_two_stream}", flush=True)

    # ---------- Iterate utterances and quantize ----------
    mel_extractor = LogMelExtractor()
    items = list(iter_librispeech_items(args.data_root, args.splits))
    print(f"[gen] processing {len(items)} utterances from {args.splits}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    targets: dict = {}
    n_total_chunks = 0
    t0 = time.time()
    for item in tqdm(items, desc="quantize"):
        mel = mel_extractor(item["audio_path"])
        mel = apply_cmvn(mel, mean, std)
        chunks = chunks_from_mel(mel, chunk_size, chunk_stride)
        if chunks.numel() == 0:
            continue
        chunks_np = chunks.numpy().astype(np.float32)
        if pca is not None:
            chunks_np = pca.transform(chunks_np)
        chunks_tensor = torch.from_numpy(chunks_np)

        result = quantize_chunks(q, chunks_tensor, device, args.quantize_batch_size)
        if is_two_stream:
            targets[item["uid"]] = {
                "z100": result["i1"],
                "z500": result["i2"],
            }
        else:
            # Flat VQ-K: real indices land in z500. z100 is a zeros placeholder
            # so the existing dataloader's `tgt["z100"]` lookup doesn't crash,
            # and SSL must be launched with --codebook-mode fine so the dummy
            # stream is never predicted against.
            z = result["single"]
            targets[item["uid"]] = {
                "z100": torch.zeros_like(z),
                "z500": z,
            }
        n_total_chunks += chunks_tensor.shape[0]

    elapsed = time.time() - t0
    print(f"[gen] quantized {len(targets)} utterances "
          f"({n_total_chunks:,} chunks) in {elapsed/60:.1f} min", flush=True)

    # ---------- Save outputs ----------
    targets_path = os.path.join(args.output_dir, "targets.pt")
    torch.save(targets, targets_path)
    print(f"[gen] wrote {targets_path}", flush=True)

    # Metadata: pretrain_ssl.py validates k_coarse/k_fine match the CLI args.
    # For two-stream we set k_coarse=K1, k_fine=K2 (so SSL runs --codebook-mode both).
    # For single-stream we set k_coarse=100 (dummy, matches z100=zeros range) and
    # k_fine=K1 (the real codebook), so SSL runs --codebook-mode fine --k-fine=K1.
    if is_two_stream:
        k_coarse, k_fine = K1, K2
    else:
        k_coarse, k_fine = 100, K1
    metadata = {
        "chunk_size": chunk_size,
        "chunk_stride": chunk_stride,
        "k_coarse": k_coarse,
        "k_fine": k_fine,
        "pca_dim": pca_dim,
        "quantizer_type": qtype,
        "quantizer_K1": K1,
        "quantizer_K2": K2,
        "is_two_stream": is_two_stream,
        "num_target_utterances": len(targets),
        "source_targets_dir": args.source_targets_dir,
        "quantizer_dir": args.quantizer_dir,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[gen] wrote metadata: k_coarse={k_coarse} k_fine={k_fine}", flush=True)

    # Copy CMVN + cluster artifacts so the existing pretrain_ssl validate
    # hook finds everything it expects.
    shutil.copy(src_cmvn_path, os.path.join(args.output_dir, "cmvn.pt"))
    shutil.copy(src_artifacts_path, os.path.join(args.output_dir, "cluster_artifacts.joblib"))
    print(f"[gen] copied cmvn.pt and cluster_artifacts.joblib", flush=True)
    print(f"[gen] done.", flush=True)


if __name__ == "__main__":
    main()
