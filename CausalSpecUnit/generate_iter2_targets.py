import argparse
import json
import os
import random
import shutil
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from CausalSpecUnit.common import TRAIN_SPLITS, strip_state_prefixes
from CausalSpecUnit.data import LogMelExtractor, apply_cmvn, iter_librispeech_items, load_cmvn
from CausalSpecUnit.model import CausalSpecUnitSSL


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="dataset/datasets/librispeech/LibriSpeech")
    p.add_argument("--splits", nargs="+", default=TRAIN_SPLITS)
    p.add_argument("--cmvn-path", type=str, required=True)
    p.add_argument("--ssl-checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs/causal_specunit/targets_iter2_c8")
    p.add_argument("--variant", type=str, default="xs", choices=["xs", "s", "sm", "m", "ml", "l"])
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--chunk-stride", type=int, default=4)
    p.add_argument("--pca-dim", type=int, default=64,
                   help="PCA output dim before k-means. Set to 0 to cluster encoder features directly.")
    p.add_argument("--k-coarse", type=int, default=100)
    p.add_argument("--k-fine", type=int, default=500)
    p.add_argument("--max-fit-frames", type=int, default=1_000_000)
    p.add_argument("--fit-frames-per-batch", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dataloader-timeout", type=int, default=120)
    p.add_argument("--target-shards", type=int, default=128)
    p.add_argument("--max-utterances", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class EncoderFeatureDataset(Dataset):
    def __init__(self, data_root, splits, cmvn_path, max_utterances=None, shuffle_items=False, seed=42):
        self.items = list(iter_librispeech_items(data_root, splits))
        if max_utterances is not None:
            self.items = self.items[:max_utterances]
        if shuffle_items:
            random.Random(seed).shuffle(self.items)
        self.mean, self.std = load_cmvn(cmvn_path)
        self.extractor = LogMelExtractor()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        mel = apply_cmvn(self.extractor(item["audio_path"]), self.mean, self.std)
        return {
            "uid": item["uid"],
            "split": item["split"],
            "mel": mel,
        }


def collate_features(batch):
    mels = nn.utils.rnn.pad_sequence([b["mel"] for b in batch], batch_first=True)
    lengths = torch.tensor([b["mel"].size(0) for b in batch], dtype=torch.long)
    return [b["uid"] for b in batch], [b["split"] for b in batch], mels, lengths


def checkpoint_file(path):
    return os.path.join(path, "checkpoint.pt") if os.path.isdir(path) else path


def load_ssl_model(checkpoint_path, variant, device):
    ckpt_path = checkpoint_file(checkpoint_path)
    state = torch.load(ckpt_path, map_location="cpu")
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    model = CausalSpecUnitSSL(variant=variant)
    missing, unexpected = model.load_state_dict(strip_state_prefixes(model_state), strict=False)
    model.to(device)
    model.eval()
    return model, missing, unexpected, state


@torch.inference_mode()
def encode_batch(model, mels, lengths, device):
    mels = mels.to(device, non_blocking=True)
    lengths = lengths.to(device, non_blocking=True)
    encoded, out_lengths = model.encoder(mels, lengths)
    return encoded.detach().float().cpu(), out_lengths.detach().cpu()


def make_loader(dataset, args):
    worker_kwargs = {}
    if args.workers > 0:
        worker_kwargs = {
            "persistent_workers": True,
            "prefetch_factor": 2,
        }
    timeout = args.dataloader_timeout if args.workers > 0 else 0
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_features,
        pin_memory=torch.cuda.is_available(),
        timeout=timeout,
        **worker_kwargs,
    )


def collect_fit_features(model, args, device):
    dataset = EncoderFeatureDataset(
        args.data_root,
        args.splits,
        args.cmvn_path,
        max_utterances=args.max_utterances,
        shuffle_items=True,
        seed=args.seed,
    )
    loader = make_loader(dataset, args)
    rng = np.random.default_rng(args.seed)
    fit_features = None
    fit_count = 0
    seen_frames = 0

    print(f"Collecting up to {args.max_fit_frames:,} iter-2 encoder frames from {len(dataset)} utterances")
    for _, _, mels, lengths in tqdm(loader, desc="fit features"):
        encoded, out_lengths = encode_batch(model, mels, lengths, device)
        rows = []
        for i in range(encoded.size(0)):
            valid = int(out_lengths[i].item())
            if valid > 0:
                rows.append(encoded[i, :valid])
        if not rows:
            continue
        rows = torch.cat(rows, dim=0).numpy().astype(np.float32, copy=False)
        seen_frames += int(rows.shape[0])

        max_rows = min(args.fit_frames_per_batch, rows.shape[0])
        if max_rows > 0 and rows.shape[0] > max_rows:
            idx = rng.choice(rows.shape[0], size=max_rows, replace=False)
            rows = rows[idx]
        else:
            rng.shuffle(rows, axis=0)

        if fit_features is None:
            fit_features = np.empty((args.max_fit_frames, rows.shape[1]), dtype=np.float32)
        take = min(args.max_fit_frames - fit_count, rows.shape[0])
        if take > 0:
            fit_features[fit_count:fit_count + take] = rows[:take]
            fit_count += take
        if fit_count >= args.max_fit_frames:
            break

    if fit_features is None or fit_count == 0:
        raise RuntimeError("No encoder frames were collected for iter-2 PCA/k-means")
    return fit_features[:fit_count], len(dataset), seen_frames


def fit_clusters(features, args):
    if args.pca_dim > 0:
        print(f"Fitting PCA ({features.shape[1]} -> {args.pca_dim}) on {features.shape[0]:,} encoder frames")
        pca = PCA(n_components=args.pca_dim, whiten=True, random_state=args.seed)
        reduced = pca.fit_transform(features)
    else:
        print(f"Skipping PCA; clustering {features.shape[1]}-dim encoder features directly")
        pca = None
        reduced = features

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
    return pca, km_coarse, km_fine


@torch.inference_mode()
def assign_targets(model, pca, km_coarse, km_fine, args, device):
    dataset = EncoderFeatureDataset(
        args.data_root,
        args.splits,
        args.cmvn_path,
        max_utterances=args.max_utterances,
        shuffle_items=False,
        seed=args.seed,
    )
    loader = make_loader(dataset, args)
    targets = {}
    hist100 = np.zeros(args.k_coarse, dtype=np.int64)
    hist500 = np.zeros(args.k_fine, dtype=np.int64)
    total_frames = 0

    print("Assigning iter-2 targets for all utterances")
    for uids, _, mels, lengths in tqdm(loader, desc="assign"):
        encoded, out_lengths = encode_batch(model, mels, lengths, device)
        feature_parts = []
        part_lengths = []
        valid_uids = []
        for i, uid in enumerate(uids):
            valid = int(out_lengths[i].item())
            if valid <= 0:
                continue
            feature_parts.append(encoded[i, :valid].numpy().astype(np.float32, copy=False))
            part_lengths.append(valid)
            valid_uids.append(uid)
        if not feature_parts:
            continue

        flat = np.concatenate(feature_parts, axis=0)
        transformed = pca.transform(flat) if pca is not None else flat
        z100_all = km_coarse.predict(transformed).astype(np.int64)
        z500_all = km_fine.predict(transformed).astype(np.int64)

        offset = 0
        for uid, length in zip(valid_uids, part_lengths):
            z100 = z100_all[offset:offset + length]
            z500 = z500_all[offset:offset + length]
            offset += length
            hist100 += np.bincount(z100, minlength=args.k_coarse)
            hist500 += np.bincount(z500, minlength=args.k_fine)
            total_frames += int(length)
            targets[uid] = {
                "z100": torch.from_numpy(z100),
                "z500": torch.from_numpy(z500),
            }
    return targets, hist100, hist500, len(dataset), total_frames


def write_target_shards(targets, output_dir, num_shards):
    if num_shards <= 0:
        return
    shards_dir = os.path.join(output_dir, "targets_shards")
    os.makedirs(shards_dir, exist_ok=True)
    uids = sorted(targets)
    num_shards = max(1, min(num_shards, len(uids)))
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
    with open(os.path.join(output_dir, "target_index.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_targets": len(uid_to_shard),
                "num_shards": num_shards,
                "shards_dir": "targets_shards",
                "uid_to_shard": uid_to_shard,
            },
            f,
        )


def main():
    args = parse_args()
    if args.max_fit_frames <= 0:
        raise ValueError("--max-fit-frames must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    print(f"Loading iter-1 SSL model from {args.ssl_checkpoint}")
    model, missing, unexpected, checkpoint_state = load_ssl_model(args.ssl_checkpoint, args.variant, device)
    print(f"Loaded model | missing={len(missing)} unexpected={len(unexpected)} device={device}")

    cmvn_out = os.path.join(args.output_dir, "cmvn.pt")
    shutil.copyfile(args.cmvn_path, cmvn_out)

    start = time.time()
    fit_features, fit_utterances, seen_fit_frames = collect_fit_features(model, args, device)
    pca, km_coarse, km_fine = fit_clusters(fit_features, args)
    targets, hist100, hist500, assign_utterances, total_frames = assign_targets(
        model, pca, km_coarse, km_fine, args, device
    )

    torch.save(targets, os.path.join(args.output_dir, "targets.pt"))
    write_target_shards(targets, args.output_dir, args.target_shards)

    joblib.dump(
        {
            "pca": pca,
            "kmeans_coarse": km_coarse,
            "kmeans_fine": km_fine,
            "feature_dim": int(fit_features.shape[1]),
            "pca_dim": args.pca_dim,
            "k_coarse": args.k_coarse,
            "k_fine": args.k_fine,
            "num_fit_frames": int(fit_features.shape[0]),
            "target_features": "ssl_encoder_iter2",
            "source_ssl_checkpoint": args.ssl_checkpoint,
        },
        os.path.join(args.output_dir, "cluster_artifacts.joblib"),
    )

    metadata = {
        "data_root": args.data_root,
        "splits": args.splits,
        "chunk_size": args.chunk_size,
        "chunk_stride": args.chunk_stride,
        "pca_dim": args.pca_dim,
        "k_coarse": args.k_coarse,
        "k_fine": args.k_fine,
        "max_fit_frames": args.max_fit_frames,
        "fit_frames_per_batch": args.fit_frames_per_batch,
        "max_utterances": args.max_utterances,
        "seed": args.seed,
        "target_features": "ssl_encoder_iter2",
        "feature_dim": int(fit_features.shape[1]),
        "num_utterances": int(assign_utterances),
        "num_target_utterances": len(targets),
        "target_shards": args.target_shards,
        "num_fit_frames": int(fit_features.shape[0]),
        "seen_fit_frames_before_cap": int(seen_fit_frames),
        "num_encoder_frames": int(total_frames),
        "source_ssl_checkpoint": args.ssl_checkpoint,
        "source_checkpoint_epoch": int(checkpoint_state.get("epoch", 0)) if isinstance(checkpoint_state, dict) else None,
        "source_checkpoint_optimizer_steps": int(checkpoint_state.get("optimizer_steps", 0)) if isinstance(checkpoint_state, dict) else None,
        "elapsed_hours": (time.time() - start) / 3600,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    npz_kwargs = dict(
        hist100=hist100,
        hist500=hist500,
        coarse_centers=km_coarse.cluster_centers_,
        fine_centers=km_fine.cluster_centers_,
    )
    if pca is not None:
        npz_kwargs["explained_variance_ratio"] = pca.explained_variance_ratio_
    np.savez(os.path.join(args.output_dir, "cluster_stats.npz"), **npz_kwargs)

    print(f"Wrote iter-2 targets to {args.output_dir}")
    print(f"Targets: {len(targets)} utterances | encoder frames: {total_frames:,}")
    print(f"CMVN copied to: {cmvn_out}")


if __name__ == "__main__":
    main()
