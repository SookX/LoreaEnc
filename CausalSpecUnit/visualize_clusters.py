import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets-dir", type=str, default="outputs/causal_specunit/targets")
    p.add_argument("--output-dir", type=str, default="outputs/causal_specunit/figures")
    p.add_argument("--top-n", type=int, default=24)
    p.add_argument("--timeline-utterances", type=int, default=8)
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def plot_hist(hist, title, path):
    order = np.argsort(hist)[::-1]
    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(len(hist)), hist[order])
    plt.title(title)
    plt.xlabel("Clusters sorted by frequency")
    plt.ylabel("Assigned chunks")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_coverage(hist, title, path):
    counts = np.sort(hist)[::-1]
    cumulative = np.cumsum(counts) / max(counts.sum(), 1)
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(counts) + 1), cumulative, linewidth=2)
    plt.title(title)
    plt.xlabel("Top-N clusters")
    plt.ylabel("Fraction of assigned chunks covered")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_pca_variance(evr, path):
    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(evr), marker="o", linewidth=1)
    plt.title("PCA cumulative explained variance")
    plt.xlabel("PCA dimensions")
    plt.ylabel("Cumulative explained variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def centers_to_chunks(pca, centers, chunk_size):
    flat = pca.inverse_transform(centers)
    return flat.reshape(flat.shape[0], chunk_size, 80)


def plot_centroid_grid(chunks, hist, title, path, top_n):
    top = np.argsort(hist)[::-1][:top_n]
    ncols = 6
    nrows = int(np.ceil(len(top) / ncols))
    plt.figure(figsize=(2.4 * ncols, 2.0 * nrows))
    vmax = np.percentile(np.abs(chunks[top]), 98)
    for i, cluster_id in enumerate(top):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.imshow(chunks[cluster_id].T, aspect="auto", origin="lower", cmap="magma", vmin=-vmax, vmax=vmax)
        ax.set_title(f"k={cluster_id}\nn={hist[cluster_id]}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_tsne(centers, hist, title, path, seed, perplexity):
    if centers.shape[0] < 4:
        return
    perplexity = min(perplexity, max(2.0, (centers.shape[0] - 1) / 3))
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed).fit_transform(centers)
    sizes = 10 + 90 * (hist / max(hist.max(), 1))
    plt.figure(figsize=(7, 6))
    plt.scatter(emb[:, 0], emb[:, 1], s=sizes, c=np.log1p(hist), cmap="viridis", alpha=0.85)
    plt.colorbar(label="log assigned chunks")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def collect_target_arrays(targets, key):
    arrays = []
    for value in targets.values():
        z = value[key].numpy() if isinstance(value[key], torch.Tensor) else np.asarray(value[key])
        if z.size > 1:
            arrays.append(z.astype(np.int64))
    return arrays


def plot_transition_matrix(target_arrays, num_clusters, title, path, top_n=40):
    hist = np.bincount(np.concatenate(target_arrays), minlength=num_clusters)
    top = np.argsort(hist)[::-1][:top_n]
    index = {cluster_id: i for i, cluster_id in enumerate(top)}
    mat = np.zeros((top_n, top_n), dtype=np.float64)
    for z in target_arrays:
        for a, b in zip(z[:-1], z[1:]):
            if a in index and b in index:
                mat[index[a], index[b]] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = np.divide(mat, np.maximum(row_sums, 1.0))
    plt.figure(figsize=(8, 7))
    plt.imshow(mat, aspect="auto", cmap="magma")
    plt.colorbar(label="transition probability")
    plt.title(title)
    plt.xlabel("next cluster, sorted by frequency")
    plt.ylabel("current cluster, sorted by frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_label_timelines(targets, key, title, path, num_utterances):
    selected = list(targets.items())[:num_utterances]
    plt.figure(figsize=(12, max(3, 0.55 * len(selected))))
    for row, (uid, value) in enumerate(selected):
        z = value[key].numpy() if isinstance(value[key], torch.Tensor) else np.asarray(value[key])
        plt.scatter(np.arange(len(z)), np.full(len(z), row), c=z, s=8, cmap="tab20", marker="s")
    plt.yticks(np.arange(len(selected)), [uid for uid, _ in selected], fontsize=7)
    plt.xlabel("target time step")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def cluster_entropy(hist):
    probs = hist / max(hist.sum(), 1)
    probs = probs[probs > 0]
    entropy = -(probs * np.log2(probs)).sum()
    return entropy, entropy / np.log2(len(hist))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    artifacts = joblib.load(os.path.join(args.targets_dir, "cluster_artifacts.joblib"))
    stats = np.load(os.path.join(args.targets_dir, "cluster_stats.npz"))
    pca = artifacts["pca"]
    chunk_size = artifacts["chunk_size"]
    hist100 = stats["hist100"]
    hist500 = stats["hist500"]
    evr = stats["explained_variance_ratio"]
    c100 = stats["coarse_centers"]
    c500 = stats["fine_centers"]
    targets = torch.load(os.path.join(args.targets_dir, "targets.pt"), map_location="cpu")

    plot_hist(hist100, "K=100 cluster usage", os.path.join(args.output_dir, "k100_usage.png"))
    plot_hist(hist500, "K=500 cluster usage", os.path.join(args.output_dir, "k500_usage.png"))
    plot_coverage(hist100, "K=100 cumulative cluster coverage", os.path.join(args.output_dir, "k100_coverage.png"))
    plot_coverage(hist500, "K=500 cumulative cluster coverage", os.path.join(args.output_dir, "k500_coverage.png"))
    plot_pca_variance(evr, os.path.join(args.output_dir, "pca_explained_variance.png"))

    chunks100 = centers_to_chunks(pca, c100, chunk_size)
    chunks500 = centers_to_chunks(pca, c500, chunk_size)
    plot_centroid_grid(chunks100, hist100, "Most frequent K=100 centroids reconstructed to log-mel chunks", os.path.join(args.output_dir, "k100_centroids.png"), args.top_n)
    plot_centroid_grid(chunks500, hist500, "Most frequent K=500 centroids reconstructed to log-mel chunks", os.path.join(args.output_dir, "k500_centroids.png"), args.top_n)
    plot_tsne(c100, hist100, "t-SNE of K=100 PCA-space centroids", os.path.join(args.output_dir, "k100_tsne.png"), args.seed, args.tsne_perplexity)
    plot_tsne(c500, hist500, "t-SNE of K=500 PCA-space centroids", os.path.join(args.output_dir, "k500_tsne.png"), args.seed, args.tsne_perplexity)
    z100_arrays = collect_target_arrays(targets, "z100")
    z500_arrays = collect_target_arrays(targets, "z500")
    plot_transition_matrix(z100_arrays, artifacts["k_coarse"], "K=100 top-cluster transition matrix", os.path.join(args.output_dir, "k100_transition_matrix.png"))
    plot_transition_matrix(z500_arrays, artifacts["k_fine"], "K=500 top-cluster transition matrix", os.path.join(args.output_dir, "k500_transition_matrix.png"))
    plot_label_timelines(targets, "z100", "K=100 target timelines for example utterances", os.path.join(args.output_dir, "k100_label_timelines.png"), args.timeline_utterances)
    plot_label_timelines(targets, "z500", "K=500 target timelines for example utterances", os.path.join(args.output_dir, "k500_label_timelines.png"), args.timeline_utterances)
    ent100, norm_ent100 = cluster_entropy(hist100)
    ent500, norm_ent500 = cluster_entropy(hist500)

    explanation = f"""# Cluster Visualization Explanation

These figures explain the from-scratch spectrogram units used by CausalSpecUnit.

## How the clusters were made

1. Audio was converted to 80-bin log-mel spectrograms.
2. A global per-frequency CMVN transform was estimated over the training splits.
3. The normalized spectrogram was cut into full-band temporal chunks of shape
   {chunk_size} x 80.
4. Each chunk was flattened to {chunk_size * 80} dimensions.
5. PCA with whitening compressed each chunk to {artifacts['pca_dim']} dimensions.
6. MiniBatchKMeans was fit twice: K={artifacts['k_coarse']} for coarse units and
   K={artifacts['k_fine']} for fine units.

Targets are generated from the clean CMVN-normalized spectrogram. SSL
pretraining later corrupts a separate copy of the spectrogram and asks the model
to recover these clean labels only at masked target positions.

## Summary Statistics

- K=100 active clusters: {(hist100 > 0).sum()} / {len(hist100)}
- K=500 active clusters: {(hist500 > 0).sum()} / {len(hist500)}
- K=100 entropy: {ent100:.2f} bits ({norm_ent100:.2%} of uniform)
- K=500 entropy: {ent500:.2f} bits ({norm_ent500:.2%} of uniform)
- PCA cumulative variance at 16 dims: {evr[:16].sum():.2%}
- PCA cumulative variance at 32 dims: {evr[:32].sum():.2%}
- PCA cumulative variance at {len(evr)} dims: {evr.sum():.2%}

## Figures

- `pca_explained_variance.png`: shows how much chunk energy/variation is kept by
  the PCA dimensions. A steep early curve means the chunk representation is
  compressible.
- `k100_usage.png` and `k500_usage.png`: show cluster assignment balance. A few
  dominant clusters mean the targets may be too generic; a flatter histogram
  means the unit inventory is used more evenly.
- `k100_centroids.png` and `k500_centroids.png`: reconstruct k-means centroids
  back through inverse PCA and display them as time-frequency log-mel chunks.
  These are prototype acoustic patterns discovered without a teacher model.
- `k100_tsne.png` and `k500_tsne.png`: show the geometry of cluster centroids in
  PCA space. Larger points are more frequent clusters.
- `k100_coverage.png` and `k500_coverage.png`: show how many clusters are needed
  to cover the target stream. A steep curve means a small set of clusters
  dominates.
- `k100_transition_matrix.png` and `k500_transition_matrix.png`: show local
  temporal structure in the discrete target stream. Strong diagonal blocks mean
  units tend to persist or transition within related acoustic regions.
- `k100_label_timelines.png` and `k500_label_timelines.png`: show discrete unit
  sequences for example utterances. These help verify that labels vary over time
  rather than collapsing to a constant unit.

## What to look for

Good targets should have:

- no severe cluster collapse,
- visually distinct centroid patterns,
- enough frequent clusters to cover common acoustic events,
- rare clusters that are not only noise/outliers.

If the usage plot is extremely skewed, reduce K, improve CMVN, use more chunks,
or lower PCA dimensionality. If centroid chunks look like noise, inspect the
log-mel extraction and PCA whitening.
"""
    save_text(os.path.join(args.output_dir, "cluster_visualization_explanation.md"), explanation)
    print(f"Wrote figures and explanation to {args.output_dir}")


if __name__ == "__main__":
    main()
