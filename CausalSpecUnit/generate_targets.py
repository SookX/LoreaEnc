import argparse
import json
import os
import random

import joblib
import numpy as np
import torch
import torchaudio
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from CausalSpecUnit.common import TRAIN_SPLITS
from CausalSpecUnit.data import LogMelExtractor, apply_cmvn, iter_librispeech_items


class MFCCExtractor:
    """39-dim MFCC extractor (13 coefficients + delta + delta-delta) at 16 kHz."""

    def __init__(self, n_mfcc: int = 13, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self._transform = None  # lazy init so it's created on first call

    def _get_transform(self, device):
        if self._transform is None:
            self._transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={"n_fft": 512, "hop_length": 160, "win_length": 400, "n_mels": 80},
            ).to(device)
        return self._transform

    def __call__(self, audio_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(0, keepdim=True)
        transform = self._get_transform(wav.device)
        mfcc = transform(wav).squeeze(0).T  # [T, n_mfcc]
        delta1 = torchaudio.functional.compute_deltas(mfcc.T).T
        delta2 = torchaudio.functional.compute_deltas(delta1.T).T
        return torch.cat([mfcc, delta1, delta2], dim=-1)  # [T, 3*n_mfcc]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="dataset/datasets/librispeech/LibriSpeech")
    p.add_argument("--splits", nargs="+", default=TRAIN_SPLITS)
    p.add_argument("--output-dir", type=str, default="outputs/causal_specunit/targets")
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--chunk-stride", type=int, default=4)
    p.add_argument("--pca-dim", type=int, default=64)
    p.add_argument("--k-coarse", type=int, default=100)
    p.add_argument("--k-fine", type=int, default=500)
    p.add_argument("--max-fit-chunks", type=int, default=1_000_000)
    p.add_argument("--n-mfcc", type=int, default=13,
                   help="Number of MFCC coefficients (targets use 3x this: coeffs+delta+delta2).")
    p.add_argument("--max-utterances", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def chunks_from_mel(mel, chunk_size, chunk_stride):
    if mel.size(0) < chunk_size:
        return torch.empty(0, chunk_size * mel.size(1))
    chunks = mel.unfold(0, chunk_size, chunk_stride)  # [N, 80, C]
    chunks = chunks.transpose(1, 2).contiguous()      # [N, C, 80]
    return chunks.view(chunks.size(0), -1)


def update_cmvn(mel, state):
    state["count"] += mel.size(0)
    state["sum"] += mel.sum(dim=0)
    state["sumsq"] += (mel * mel).sum(dim=0)


def finalize_cmvn(state):
    mean = state["sum"] / max(state["count"], 1)
    var = state["sumsq"] / max(state["count"], 1) - mean * mean
    std = torch.sqrt(var.clamp_min(1e-5))
    return mean, std


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    items = list(iter_librispeech_items(args.data_root, args.splits))
    if args.max_utterances is not None:
        items = items[:args.max_utterances]

    mel_extractor = LogMelExtractor()
    mfcc_extractor = MFCCExtractor(n_mfcc=args.n_mfcc)

    cmvn_state = {
        "count": 0,
        "sum": torch.zeros(80),
        "sumsq": torch.zeros(80),
    }

    print(f"Computing global CMVN (log-mel) from {len(items)} utterances")
    for item in tqdm(items, desc="cmvn"):
        mel = mel_extractor(item["audio_path"])
        update_cmvn(mel, cmvn_state)
    mean, std = finalize_cmvn(cmvn_state)
    cmvn_path = os.path.join(args.output_dir, "cmvn.pt")
    torch.save({"mean": mean, "std": std}, cmvn_path)

    fit_chunks = []
    seen = 0
    print(f"Collecting up to {args.max_fit_chunks:,} MFCC chunks for PCA/k-means")
    for item in tqdm(items, desc="sample chunks"):
        mfcc = mfcc_extractor(item["audio_path"])
        chunks = chunks_from_mel(mfcc, args.chunk_size, args.chunk_stride)
        for chunk in chunks:
            seen += 1
            if len(fit_chunks) < args.max_fit_chunks:
                fit_chunks.append(chunk.numpy())
            else:
                j = random.randrange(seen)
                if j < args.max_fit_chunks:
                    fit_chunks[j] = chunk.numpy()

    fit_chunks = np.asarray(fit_chunks, dtype=np.float32)
    print(f"Fitting PCA on {fit_chunks.shape[0]:,} chunks of dim {fit_chunks.shape[1]}")
    pca = PCA(n_components=args.pca_dim, whiten=True, random_state=args.seed)
    reduced = pca.fit_transform(fit_chunks)

    print(f"Fitting MiniBatchKMeans K={args.k_coarse}")
    km_coarse = MiniBatchKMeans(
        n_clusters=args.k_coarse,
        batch_size=8192,
        n_init="auto",
        max_iter=300,
        random_state=args.seed,
        verbose=0,
    )
    km_coarse.fit(reduced)

    print(f"Fitting MiniBatchKMeans K={args.k_fine}")
    km_fine = MiniBatchKMeans(
        n_clusters=args.k_fine,
        batch_size=8192,
        n_init="auto",
        max_iter=300,
        random_state=args.seed + 1,
        verbose=0,
    )
    km_fine.fit(reduced)

    joblib.dump(
        {
            "pca": pca,
            "kmeans_coarse": km_coarse,
            "kmeans_fine": km_fine,
            "chunk_size": args.chunk_size,
            "chunk_stride": args.chunk_stride,
            "pca_dim": args.pca_dim,
            "k_coarse": args.k_coarse,
            "k_fine": args.k_fine,
        },
        os.path.join(args.output_dir, "cluster_artifacts.joblib"),
    )

    targets = {}
    hist100 = np.zeros(args.k_coarse, dtype=np.int64)
    hist500 = np.zeros(args.k_fine, dtype=np.int64)
    print("Assigning targets for all utterances (MFCC features)")
    for item in tqdm(items, desc="assign"):
        mfcc = mfcc_extractor(item["audio_path"])
        chunks = chunks_from_mel(mfcc, args.chunk_size, args.chunk_stride)
        if chunks.numel() == 0:
            continue
        y = pca.transform(chunks.numpy().astype(np.float32))
        z100 = km_coarse.predict(y).astype(np.int64)
        z500 = km_fine.predict(y).astype(np.int64)
        hist100 += np.bincount(z100, minlength=args.k_coarse)
        hist500 += np.bincount(z500, minlength=args.k_fine)
        targets[item["uid"]] = {
            "z100": torch.from_numpy(z100),
            "z500": torch.from_numpy(z500),
        }

    torch.save(targets, os.path.join(args.output_dir, "targets.pt"))
    metadata = {
        "data_root": args.data_root,
        "splits": args.splits,
        "chunk_size": args.chunk_size,
        "chunk_stride": args.chunk_stride,
        "pca_dim": args.pca_dim,
        "k_coarse": args.k_coarse,
        "k_fine": args.k_fine,
        "max_fit_chunks": args.max_fit_chunks,
        "max_utterances": args.max_utterances,
        "seed": args.seed,
        "target_features": "mfcc",
        "n_mfcc": args.n_mfcc,
        "mfcc_dim": args.n_mfcc * 3,
        "num_utterances": len(items),
        "num_target_utterances": len(targets),
        "num_fit_chunks": int(fit_chunks.shape[0]),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    np.savez(
        os.path.join(args.output_dir, "cluster_stats.npz"),
        hist100=hist100,
        hist500=hist500,
        explained_variance_ratio=pca.explained_variance_ratio_,
        coarse_centers=km_coarse.cluster_centers_,
        fine_centers=km_fine.cluster_centers_,
    )
    print(f"Wrote targets and artifacts to {args.output_dir}")
    print(f"CMVN: {cmvn_path}")


if __name__ == "__main__":
    main()
