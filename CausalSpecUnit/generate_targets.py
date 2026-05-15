import argparse
import json
import os
import random

import joblib
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from CausalSpecUnit.common import TRAIN_SPLITS
from CausalSpecUnit.data import LogMelExtractor, apply_cmvn, iter_librispeech_items


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="dataset/datasets/librispeech/LibriSpeech")
    p.add_argument("--splits", nargs="+", default=TRAIN_SPLITS)
    p.add_argument("--output-dir", type=str, default="outputs/causal_specunit/targets")
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--chunk-stride", type=int, default=4)
    p.add_argument("--pca-dim", type=int, default=64,
                   help="PCA output dim. Set to 0 to skip PCA.")
    p.add_argument("--k-coarse", type=int, default=100)
    p.add_argument("--k-fine", type=int, default=500)
    p.add_argument("--max-fit-chunks", type=int, default=1_000_000)
    p.add_argument("--max-utterances", type=int, default=None)
    p.add_argument("--target-shards", type=int, default=64,
                   help="Also write sharded target files for distributed training startup.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reuse-artifacts", type=str, default=None,
                   help="Path to an existing targets dir. Reuses its cluster_artifacts.joblib and cmvn.pt, "
                        "skipping CMVN computation and PCA/k-means fitting. Only re-assigns targets with "
                        "the new chunk-size/stride.")
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

    cmvn_path = os.path.join(args.output_dir, "cmvn.pt")
    fit_chunk_count = None

    if args.reuse_artifacts:
        print(f"Reusing CMVN and cluster artifacts from {args.reuse_artifacts}")
        artifacts_path = os.path.join(args.reuse_artifacts, "cluster_artifacts.joblib")
        cmvn_src = os.path.join(args.reuse_artifacts, "cmvn.pt")
        if not os.path.isfile(artifacts_path):
            raise FileNotFoundError(f"Missing cluster_artifacts.joblib in {args.reuse_artifacts}")
        if not os.path.isfile(cmvn_src):
            raise FileNotFoundError(f"Missing cmvn.pt in {args.reuse_artifacts}")
        artifacts = joblib.load(artifacts_path)
        pca = artifacts["pca"]
        km_coarse = artifacts["kmeans_coarse"]
        km_fine = artifacts["kmeans_fine"]
        cmvn = torch.load(cmvn_src, map_location="cpu")
        mean, std = cmvn["mean"].float(), cmvn["std"].float()
        torch.save({"mean": mean, "std": std}, cmvn_path)
        fit_chunk_count = int(artifacts.get("num_fit_chunks", 0))
        print(f"Loaded PCA ({artifacts.get('pca_dim')}d) + KMeans "
              f"(k_coarse={artifacts.get('k_coarse')}, k_fine={artifacts.get('k_fine')})")
        print(f"Re-assigning with chunk_size={args.chunk_size}, chunk_stride={args.chunk_stride}")
    else:
        if os.path.isfile(cmvn_path):
            print(f"Found existing CMVN, resuming from {cmvn_path}")
            cmvn = torch.load(cmvn_path, map_location="cpu")
            mean, std = cmvn["mean"].float(), cmvn["std"].float().clamp_min(1e-5)
        else:
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
            torch.save({"mean": mean, "std": std}, cmvn_path)

        fit_chunks = []
        seen = 0
        print(f"Collecting up to {args.max_fit_chunks:,} clean log-mel chunks for PCA/k-means")
        for item in tqdm(items, desc="sample chunks"):
            mel = apply_cmvn(mel_extractor(item["audio_path"]), mean, std)
            chunks = chunks_from_mel(mel, args.chunk_size, args.chunk_stride)
            for chunk in chunks:
                seen += 1
                if len(fit_chunks) < args.max_fit_chunks:
                    fit_chunks.append(chunk.numpy())
                else:
                    j = random.randrange(seen)
                    if j < args.max_fit_chunks:
                        fit_chunks[j] = chunk.numpy()

        fit_chunks = np.asarray(fit_chunks, dtype=np.float32)
        fit_chunk_count = int(fit_chunks.shape[0])
        if args.pca_dim > 0:
            print(f"Fitting PCA ({fit_chunks.shape[1]} -> {args.pca_dim} dims) on {fit_chunks.shape[0]:,} chunks")
            pca = PCA(n_components=args.pca_dim, whiten=True, random_state=args.seed)
            reduced = pca.fit_transform(fit_chunks)
        else:
            print(f"Skipping PCA — clustering directly on {fit_chunks.shape[1]}-dim features")
            pca = None
            reduced = fit_chunks

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
                "num_fit_chunks": fit_chunk_count,
            },
            os.path.join(args.output_dir, "cluster_artifacts.joblib"),
        )

    targets = {}
    hist100 = np.zeros(args.k_coarse, dtype=np.int64)
    hist500 = np.zeros(args.k_fine, dtype=np.int64)
    print("Assigning targets for all utterances (clean CMVN log-mel features)")
    for item in tqdm(items, desc="assign"):
        mel = apply_cmvn(mel_extractor(item["audio_path"]), mean, std)
        chunks = chunks_from_mel(mel, args.chunk_size, args.chunk_stride)
        if chunks.numel() == 0:
            continue
        y = chunks.numpy().astype(np.float32)
        if pca is not None:
            y = pca.transform(y)
        z100 = km_coarse.predict(y).astype(np.int64)
        z500 = km_fine.predict(y).astype(np.int64)
        hist100 += np.bincount(z100, minlength=args.k_coarse)
        hist500 += np.bincount(z500, minlength=args.k_fine)
        targets[item["uid"]] = {
            "z100": torch.from_numpy(z100),
            "z500": torch.from_numpy(z500),
        }

    torch.save(targets, os.path.join(args.output_dir, "targets.pt"))
    if args.target_shards > 0:
        shards_dir = os.path.join(args.output_dir, "targets_shards")
        os.makedirs(shards_dir, exist_ok=True)
        uids = sorted(targets)
        num_shards = max(1, min(args.target_shards, len(uids)))
        shard_size = (len(uids) + num_shards - 1) // num_shards
        uid_to_shard = {}
        for shard_id in tqdm(range(num_shards), desc="write target shards"):
            start = shard_id * shard_size
            end = min(start + shard_size, len(uids))
            shard_uids = uids[start:end]
            if not shard_uids:
                continue
            shard_name = f"targets_{shard_id:04d}.pt"
            torch.save({uid: targets[uid] for uid in shard_uids}, os.path.join(shards_dir, shard_name))
            for uid in shard_uids:
                uid_to_shard[uid] = shard_name
        with open(os.path.join(args.output_dir, "target_index.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_targets": len(uid_to_shard),
                    "num_shards": num_shards,
                    "shards_dir": "targets_shards",
                    "uid_to_shard": uid_to_shard,
                },
                f,
            )
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
        "target_features": "cmvn_log_mel",
        "feature_dim": 80,
        "num_utterances": len(items),
        "num_target_utterances": len(targets),
        "target_shards": args.target_shards,
        "num_fit_chunks": int(fit_chunk_count or 0),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    npz_kwargs = dict(
        hist100=hist100,
        hist500=hist500,
        coarse_centers=km_coarse.cluster_centers_,
        fine_centers=km_fine.cluster_centers_,
    )
    if pca is not None:
        npz_kwargs["explained_variance_ratio"] = pca.explained_variance_ratio_
    np.savez(os.path.join(args.output_dir, "cluster_stats.npz"), **npz_kwargs)
    print(f"Wrote targets and artifacts to {args.output_dir}")
    print(f"CMVN: {cmvn_path}")


if __name__ == "__main__":
    main()
