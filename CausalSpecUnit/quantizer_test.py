"""Sanity tests for CausalSpecUnit/quantizer.py. Run with:

    python -m CausalSpecUnit.quantizer_test

Assert-based and dependency-light so the same script works on a login
node without a pytest install. Should complete in < 30 s on CPU.
"""

import torch
import torch.nn.functional as F

from CausalSpecUnit.quantizer import VQEMA, RVQ, ParallelVQ


# ---------------------------------------------------------------------------
# VQEMA
# ---------------------------------------------------------------------------

def test_vqema_forward_shapes_and_dtypes():
    vq = VQEMA(dim=64, K=600)
    x = torch.randn(32, 64)
    x_q, indices, commit, ppl = vq(x)
    assert x_q.shape == (32, 64)
    assert indices.shape == (32,)
    assert indices.dtype == torch.int64
    assert 0 <= int(indices.min()) and int(indices.max()) < 600
    assert commit.dim() == 0
    assert ppl.dim() == 0
    print("OK  test_vqema_forward_shapes_and_dtypes")


def test_vqema_init_from_kmeans():
    vq = VQEMA(dim=8, K=16)
    sample = torch.randn(2000, 8)
    vq.init_from_kmeans(sample)
    # Codebook must not be all-zero
    assert vq.codebook.abs().sum() > 0
    # Perplexity on the training sample should be a healthy fraction of K
    _, _, _, ppl = vq(sample)
    assert ppl.item() > 4.0, f"ppl after init = {ppl.item():.2f}, expected > 4 for K=16"
    print(f"OK  test_vqema_init_from_kmeans (ppl_post_init={ppl.item():.2f}/16)")


def test_vqema_codebook_tracks_data_with_ema():
    """K-means init + EMA updates on K-cluster data should yield (a) a moved
    codebook (EMA does something), and (b) high perplexity (codes are
    used uniformly). We don't require Hungarian center-recovery — k-means
    init usually gets us there and EMA only refines."""
    torch.manual_seed(0)
    K = 8
    centers = torch.randn(K, 4) * 5.0
    vq = VQEMA(dim=4, K=K, decay=0.9)

    def sample_batch(n=256):
        idx = torch.randint(0, K, (n,))
        return centers[idx] + 0.05 * torch.randn(n, 4)

    vq.init_from_kmeans(sample_batch(10000))
    cb_init = vq.codebook.clone()

    for _ in range(200):
        x = sample_batch()
        _, indices, _, _ = vq(x)
        vq.update_codebook(x, indices)

    _, _, _, ppl = vq(sample_batch(2000))
    assert ppl.item() > 0.7 * K, f"ppl post-training = {ppl.item():.2f}, expected > {0.7 * K:.1f}"
    delta = (vq.codebook - cb_init).norm().item()
    assert delta > 1e-3, f"codebook didn't move at all from kmeans init, delta={delta:.6f}"
    print(f"OK  test_vqema_codebook_tracks_data_with_ema (ppl={ppl.item():.2f}/{K}, delta={delta:.3f})")


def test_vqema_dead_code_revival():
    vq = VQEMA(dim=4, K=8, decay=0.5)
    sample = torch.randn(1000, 4)
    vq.init_from_kmeans(sample)
    # Force two codes to be dead
    vq.ema_N[3] = 0.0
    vq.ema_N[5] = 0.0
    n_revived = vq.revive_dead_codes(sample, threshold=0.001)
    assert n_revived >= 2, f"expected at least 2 revivals, got {n_revived}"
    # After revival, the previously-dead codes should have nonzero ema_N
    assert vq.ema_N[3] > 0 and vq.ema_N[5] > 0
    print(f"OK  test_vqema_dead_code_revival (revived={n_revived})")


def test_vqema_active_codes_metric():
    vq = VQEMA(dim=4, K=10)
    vq.ema_N.fill_(1.0)
    vq.ema_N[3:6] = 0.0  # make 3 dead
    n_active = vq.active_codes(threshold=0.001)
    assert n_active == 7, f"expected 7 active, got {n_active}"
    print(f"OK  test_vqema_active_codes_metric (active={n_active}/10)")


# ---------------------------------------------------------------------------
# RVQ
# ---------------------------------------------------------------------------

def test_rvq_init_and_residual_shrinks():
    rvq = RVQ(dim=64, K1=100, K2=500)
    sample = torch.randn(3000, 64)
    rvq.init_from_kmeans(sample)
    with torch.no_grad():
        d1 = rvq.q1._distances(sample)
        i1 = d1.argmin(dim=-1)
        x_q1 = F.embedding(i1, rvq.q1.codebook)
        residual_norm = (sample - x_q1).norm(dim=-1).mean()
        sample_norm = sample.norm(dim=-1).mean()
    assert residual_norm < sample_norm
    print(f"OK  test_rvq_init_and_residual_shrinks (|x|={sample_norm:.2f} -> |x-q1|={residual_norm:.2f})")


def test_rvq_forward_and_index_ranges():
    rvq = RVQ(dim=64, K1=100, K2=500)
    sample = torch.randn(2000, 64)
    rvq.init_from_kmeans(sample)
    x = torch.randn(32, 64)
    x_q, (i1, i2), _commit, (ppl1, ppl2) = rvq(x)
    assert x_q.shape == (32, 64)
    assert i1.shape == (32,) and i2.shape == (32,)
    assert int(i1.max()) < 100 and int(i2.max()) < 500
    assert ppl1.item() > 0 and ppl2.item() > 0
    print(f"OK  test_rvq_forward_and_index_ranges (ppl1={ppl1.item():.1f}/100, ppl2={ppl2.item():.1f}/500)")


def test_rvq_reconstruction_better_than_q1_alone():
    """The two-level sum x_q1+x_q2 should reconstruct x at least as well as
    x_q1 alone, after enough training."""
    rvq = RVQ(dim=8, K1=4, K2=4, decay=0.9)
    torch.manual_seed(1)
    sample = torch.randn(2000, 8) * 3.0
    rvq.init_from_kmeans(sample)

    for _ in range(200):
        x = torch.randn(256, 8) * 3.0
        _x_q, (i1, i2), _, _ = rvq(x)
        rvq.update_codebooks(x, (i1, i2))

    x_test = torch.randn(1000, 8) * 3.0
    with torch.no_grad():
        d1 = rvq.q1._distances(x_test)
        i1 = d1.argmin(dim=-1)
        x_q1 = F.embedding(i1, rvq.q1.codebook)
        err_q1 = (x_test - x_q1).norm(dim=-1).mean()
        x_q_full, _, _, _ = rvq(x_test)
        err_full = (x_test - x_q_full).norm(dim=-1).mean()
    assert err_full <= err_q1 + 1e-3, (
        f"two-level err {err_full:.3f} should not exceed one-level err {err_q1:.3f}"
    )
    print(f"OK  test_rvq_reconstruction_better_than_q1_alone (|x-q1|={err_q1:.3f}, |x-(q1+q2)|={err_full:.3f})")


# ---------------------------------------------------------------------------
# ParallelVQ
# ---------------------------------------------------------------------------

def test_parallel_vq_forward_and_index_ranges():
    pvq = ParallelVQ(dim=64, K1=100, K2=500)
    sample = torch.randn(2000, 64)
    pvq.init_from_kmeans(sample)
    x = torch.randn(32, 64)
    _, (i1, i2), _commit, (ppl1, ppl2) = pvq(x)
    assert i1.shape == (32,) and i2.shape == (32,)
    assert int(i1.max()) < 100 and int(i2.max()) < 500
    print(f"OK  test_parallel_vq_forward_and_index_ranges (ppl1={ppl1.item():.1f}/100, ppl2={ppl2.item():.1f}/500)")


def test_parallel_vq_indices_can_differ():
    """The two VQs should make different assignments when K1 != K2 (the
    finer-grained codebook should disagree with the coarser one)."""
    pvq = ParallelVQ(dim=8, K1=4, K2=16, decay=0.9)
    torch.manual_seed(2)
    sample = torch.randn(2000, 8) * 2.0
    pvq.init_from_kmeans(sample)
    x = torch.randn(64, 8) * 2.0
    _, (i1, i2), _, _ = pvq(x)
    # We don't require disagreement at every position, but at least one.
    assert (i1 != i2).any(), "parallel VQs gave identical assignments — too aligned"
    print("OK  test_parallel_vq_indices_can_differ")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_vqema_forward_shapes_and_dtypes,
        test_vqema_init_from_kmeans,
        test_vqema_codebook_tracks_data_with_ema,
        test_vqema_dead_code_revival,
        test_vqema_active_codes_metric,
        test_rvq_init_and_residual_shrinks,
        test_rvq_forward_and_index_ranges,
        test_rvq_reconstruction_better_than_q1_alone,
        test_parallel_vq_forward_and_index_ranges,
        test_parallel_vq_indices_can_differ,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERR  {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print()
    if failed:
        print(f"FAILED: {failed}/{len(tests)} tests")
        raise SystemExit(1)
    print(f"ALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
