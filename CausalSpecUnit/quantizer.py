"""Learned vector quantizers for the E1 extension.

Three classes, sharing a common EMA-update codebook:

* VQEMA: a flat learned VQ with K codes and EMA codebook updates
  (VQ-VAE-2 style). Used for the C7 flat-VQ-600 cell and as the building
  block for the two-stream variants below.

* RVQ: two-level residual VQ. Level 1 quantizes the input, level 2
  quantizes the residual. Used for the C8 RVQ-100+500 cell.

* ParallelVQ: two independent VQs operating on the same input. Used for
  the C8b parallel-VQ-100+500 cell, which is the strict learned analogue
  of the parallel k-means dual codebook.

All three operate on the same whitened PCA-64 chunks that the k-means
pipeline uses; only the assignment mechanism varies. Codebook training is
EMA-only — commitment loss is computed and returned for logging but does
not drive any backpropagation step in our usage (there is no upstream
encoder to update; the input is fixed PCA-64).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQEMA(nn.Module):
    """VQ-VAE-style vector quantizer with EMA codebook updates.

    The codebook is a buffer (not a Parameter): it is updated by accumulated
    assignment statistics, not by gradient descent. Forward computes nearest
    indices, the quantized output (via straight-through estimator so any
    upstream encoder can backprop), the commitment loss, and the codebook
    perplexity over the batch.
    """

    def __init__(
        self,
        dim: int,
        K: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.K = K
        self.beta = beta
        self.decay = decay
        self.eps = eps

        # Codebook: (K, D). Buffer because EMA-updated, not gradient-trained.
        # Initialised small-random; overwritten by init_from_kmeans() before
        # training so the VQ starts from the same place as the k-means baseline.
        self.register_buffer("codebook", torch.randn(K, dim) * 0.01)

        # EMA accumulators for VQ-VAE-2 codebook updates.
        # ema_N tracks per-code usage; ema_m tracks per-code running sum of
        # assigned input vectors. The codebook is computed as ema_m / ema_N
        # after each update with Laplace smoothing.
        self.register_buffer("ema_N", torch.zeros(K))
        self.register_buffer("ema_m", torch.zeros(K, dim))

    @torch.no_grad()
    def init_from_kmeans(self, sample: torch.Tensor) -> None:
        """Initialise the codebook from MiniBatch k-means on a vector sample.

        sample: (N, D) tensor with N >= K. Detaches and moves to CPU for the
        sklearn fit, then copies centroids back into the buffer.
        """
        from sklearn.cluster import MiniBatchKMeans

        N = sample.shape[0]
        if N < self.K:
            raise ValueError(f"need at least K={self.K} samples for init, got {N}")
        km = MiniBatchKMeans(
            n_clusters=self.K,
            batch_size=8192,
            n_init="auto",
            max_iter=300,
            random_state=0,
        )
        km.fit(sample.detach().cpu().numpy())
        self.codebook.copy_(torch.from_numpy(km.cluster_centers_).to(self.codebook))

        # Seed EMA accumulators consistently so the very first update is stable.
        # We treat the k-means init as "one effective pass" of uniform usage.
        self.ema_N.fill_(1.0)
        self.ema_m.copy_(self.codebook.clone())

    def _distances(self, x: torch.Tensor) -> torch.Tensor:
        """Squared L2 distances from each x to each codebook entry. (B, K)."""
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 <x, c>
        x_norm = (x ** 2).sum(dim=-1, keepdim=True)        # (B, 1)
        c_norm = (self.codebook ** 2).sum(dim=-1)          # (K,)
        cross = x @ self.codebook.t()                      # (B, K)
        return x_norm + c_norm.unsqueeze(0) - 2.0 * cross

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize x.

        x: (B, D).
        Returns (x_q_ste, indices, commitment_loss, perplexity):
            x_q_ste: straight-through quantized output, (B, D).
            indices: int64, (B,), in [0, K).
            commitment_loss: scalar = beta * MSE(x_q.detach(), x).
            perplexity: scalar exp(H(p)) over batch assignments.
        """
        d = self._distances(x)
        indices = d.argmin(dim=-1)
        x_q = F.embedding(indices, self.codebook)

        # Commitment loss. In our offline-target-extraction usage there is no
        # upstream encoder, so this value is logged but never .backward()'d.
        commitment_loss = self.beta * F.mse_loss(x_q.detach(), x)

        # Straight-through estimator: identity gradient from x_q to x. Costs
        # nothing if no upstream module needs gradients.
        x_q_ste = x + (x_q - x).detach()

        # Perplexity over batch. Healthy: ~K. Collapsed: ~1.
        onehot = F.one_hot(indices, num_classes=self.K).float()
        probs = onehot.mean(dim=0)
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())

        return x_q_ste, indices, commitment_loss, perplexity

    @torch.no_grad()
    def update_codebook(self, x: torch.Tensor, indices: torch.Tensor) -> None:
        """One EMA update step. Call after forward() with the same x and
        the returned indices.
        """
        onehot = F.one_hot(indices, num_classes=self.K).float()
        batch_N = onehot.sum(dim=0)                # (K,)
        batch_m = onehot.t() @ x                   # (K, D)

        self.ema_N.mul_(self.decay).add_(batch_N, alpha=1.0 - self.decay)
        self.ema_m.mul_(self.decay).add_(batch_m, alpha=1.0 - self.decay)

        # Laplace smoothing of ema_N to avoid division by zero for unused codes.
        n = self.ema_N.sum()
        smoothed_N = (self.ema_N + self.eps) / (n + self.K * self.eps) * n

        self.codebook.copy_(self.ema_m / smoothed_N.unsqueeze(-1))

    @torch.no_grad()
    def revive_dead_codes(self, x: torch.Tensor, threshold: float = 0.001) -> int:
        """Re-initialise codes with usage < threshold * mean_usage by
        sampling fresh replacement vectors from x.

        Returns the number of revived codes.
        """
        mean_usage = self.ema_N.mean().clamp_min(1e-8)
        dead_mask = self.ema_N < threshold * mean_usage
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0 or x.size(0) == 0:
            return 0

        idx = torch.randint(0, x.size(0), (n_dead,), device=x.device)
        replacements = x[idx]
        dead_indices = torch.nonzero(dead_mask, as_tuple=True)[0]
        self.codebook[dead_indices] = replacements
        # Re-seed EMA accumulators so revived codes have a fighting chance.
        self.ema_N[dead_indices] = mean_usage
        self.ema_m[dead_indices] = replacements * mean_usage
        return n_dead

    def active_codes(self, threshold: float = 0.001) -> int:
        """Count codes with usage above threshold * mean. For monitoring."""
        with torch.no_grad():
            mean_usage = self.ema_N.mean().clamp_min(1e-8)
            return int((self.ema_N >= threshold * mean_usage).sum().item())


class RVQ(nn.Module):
    """Residual VQ: level 1 quantizes the input, level 2 quantizes the
    residual (input - level-1 reconstruction). Produces two index streams
    (z1, z2) that map onto the existing (z100, z500) target schema.
    """

    def __init__(
        self,
        dim: int,
        K1: int = 100,
        K2: int = 500,
        beta: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.q1 = VQEMA(dim, K1, beta=beta, decay=decay)
        self.q2 = VQEMA(dim, K2, beta=beta, decay=decay)

    @torch.no_grad()
    def init_from_kmeans(self, sample: torch.Tensor) -> None:
        """Init q1 on the input sample, then q2 on the residual after q1."""
        self.q1.init_from_kmeans(sample)
        d = self.q1._distances(sample)
        i1 = d.argmin(dim=-1)
        x_q1 = F.embedding(i1, self.q1.codebook)
        self.q2.init_from_kmeans(sample - x_q1)

    def forward(self, x: torch.Tensor):
        x_q1, i1, c1, ppl1 = self.q1(x)
        # Detach so q2's commitment loss never leaks gradients into q1; the
        # codebooks are EMA-only anyway, but this keeps the autograd graph clean.
        residual = x - x_q1.detach()
        x_q2, i2, c2, ppl2 = self.q2(residual)
        x_q = x_q1 + x_q2
        return x_q, (i1, i2), c1 + c2, (ppl1, ppl2)

    @torch.no_grad()
    def update_codebooks(self, x: torch.Tensor, indices_pair):
        i1, i2 = indices_pair
        self.q1.update_codebook(x, i1)
        # Recompute residual against the JUST-UPDATED q1 codebook so that
        # q2's update reflects q1's new state. Iterating in this order keeps
        # the level-2 codebook consistent with the level-1 it sits below.
        x_q1 = F.embedding(i1, self.q1.codebook)
        self.q2.update_codebook(x - x_q1, i2)

    @torch.no_grad()
    def revive_dead_codes(self, x: torch.Tensor, threshold: float = 0.001):
        n1 = self.q1.revive_dead_codes(x, threshold)
        d = self.q1._distances(x)
        i1 = d.argmin(dim=-1)
        residual = x - F.embedding(i1, self.q1.codebook)
        n2 = self.q2.revive_dead_codes(residual, threshold)
        return n1, n2


class ParallelVQ(nn.Module):
    """Two independent VQs operating on the same input. The strict learned
    analogue of the k-means dual codebook (which is also parallel rather
    than residual). Produces (z1, z2) -> (z100, z500) in the target schema.
    """

    def __init__(
        self,
        dim: int,
        K1: int = 100,
        K2: int = 500,
        beta: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.q1 = VQEMA(dim, K1, beta=beta, decay=decay)
        self.q2 = VQEMA(dim, K2, beta=beta, decay=decay)

    @torch.no_grad()
    def init_from_kmeans(self, sample: torch.Tensor) -> None:
        self.q1.init_from_kmeans(sample)
        self.q2.init_from_kmeans(sample)

    def forward(self, x: torch.Tensor):
        x_q1, i1, c1, ppl1 = self.q1(x)
        x_q2, i2, c2, ppl2 = self.q2(x)
        # No single canonical reconstruction. Return q1's output as the
        # nominal x_q so the API matches RVQ; downstream only needs indices.
        return x_q1, (i1, i2), c1 + c2, (ppl1, ppl2)

    @torch.no_grad()
    def update_codebooks(self, x: torch.Tensor, indices_pair):
        i1, i2 = indices_pair
        self.q1.update_codebook(x, i1)
        self.q2.update_codebook(x, i2)

    @torch.no_grad()
    def revive_dead_codes(self, x: torch.Tensor, threshold: float = 0.001):
        n1 = self.q1.revive_dead_codes(x, threshold)
        n2 = self.q2.revive_dead_codes(x, threshold)
        return n1, n2
