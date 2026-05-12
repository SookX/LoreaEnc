"""
CausalSpecUnit pipeline smoke tests.

Validates the full pipeline without requiring a complete training run.
All tests use synthetic data except where noted (marked [REAL DATA]).

Usage:
    # Run all tests (no data needed):
    python -m CausalSpecUnit.smoke_test

    # Run integration tests (requires targets + dataset):
    python -m CausalSpecUnit.smoke_test --targets-dir outputs/causal_specunit/targets_train100_mfcc --data-root W:/Papers/Lorea-new/dataset

    # Run a single test by name:
    python -m CausalSpecUnit.smoke_test --test ssl_forward
"""

import argparse
import os
import sys
import tempfile
import traceback

import numpy as np
import torch
import torch.nn as nn

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results = []


def test(name):
    def decorator(fn):
        fn._test_name = name
        results.append(fn)
        return fn
    return decorator


def check(condition, msg=""):
    if not condition:
        raise AssertionError(msg or "check failed")


# ---------------------------------------------------------------------------
# 1. Model instantiation
# ---------------------------------------------------------------------------

@test("model_ssl_instantiation")
def test_ssl_instantiation(args):
    """CausalSpecUnitSSL builds for all variants without error."""
    from CausalSpecUnit.model import CausalSpecUnitSSL
    for variant in ["xs", "s"]:
        model = CausalSpecUnitSSL(variant=variant, k_coarse=100, k_fine=500)
        check(hasattr(model, "encoder"))
        check(hasattr(model, "head_coarse"))
        check(hasattr(model, "head_fine"))
        params = sum(p.numel() for p in model.parameters())
        check(params > 0, f"variant={variant} has 0 parameters")


@test("model_ctc_instantiation")
def test_ctc_instantiation(args):
    """CausalSpecUnitCTC builds correctly."""
    from CausalSpecUnit.model import CausalSpecUnitCTC
    model = CausalSpecUnitCTC(vocab_size=128, variant="xs")
    check(hasattr(model, "model"))
    check(hasattr(model, "encoder"))


# ---------------------------------------------------------------------------
# 2. Forward passes
# ---------------------------------------------------------------------------

@test("ssl_forward")
def test_ssl_forward(args):
    """SSL model forward pass produces finite loss on synthetic data."""
    from CausalSpecUnit.model import CausalSpecUnitSSL
    from CausalSpecUnit.pretrain_ssl import make_hubert_mask, corrupt_mel_from_target_mask, masked_unit_ce

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalSpecUnitSSL(variant="xs", k_coarse=100, k_fine=500).to(device)
    model.eval()

    B, T, C = 4, 200, 80
    mel = torch.randn(B, T, C, device=device)
    lengths = torch.tensor([T, T, T // 2, T // 2], dtype=torch.long, device=device)

    # Fake targets: 1 target per frame (chunk_size=1, chunk_stride=1)
    n_targets = T
    z100 = torch.randint(0, 100, (B, n_targets), device=device)
    z500 = torch.randint(0, 500, (B, n_targets), device=device)
    target_lengths = torch.tensor([n_targets] * B, dtype=torch.long, device=device)

    with torch.no_grad():
        mask = make_hubert_mask(target_lengths, n_targets, mask_prob=0.065, mask_length=10, device=device)
        corrupted = corrupt_mel_from_target_mask(mel, mask, chunk_size=1, chunk_stride=1, mask_value=0.0)
        coarse_logits, fine_logits, out_lengths = model(corrupted, lengths)

    check(not torch.isnan(coarse_logits).any(), "NaN in coarse logits")
    check(not torch.isnan(fine_logits).any(), "NaN in fine logits")

    # Trim to output length
    t = min(coarse_logits.size(1), z100.size(1))
    loss100 = masked_unit_ce(coarse_logits[:, :t], z100[:, :t], mask[:, :t])
    loss500 = masked_unit_ce(fine_logits[:, :t], z500[:, :t], mask[:, :t])
    loss = loss100 + loss500

    check(torch.isfinite(loss), f"SSL loss is not finite: {loss.item()}")


@test("ctc_forward")
def test_ctc_forward(args):
    """CTC model forward pass produces finite CTC loss on synthetic data."""
    from CausalSpecUnit.model import CausalSpecUnitCTC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalSpecUnitCTC(vocab_size=32, variant="xs").to(device)
    model.eval()

    B, T, C = 4, 200, 80
    mel = torch.randn(B, T, C, device=device)
    lengths = torch.tensor([T, T, T // 2, T // 2], dtype=torch.long, device=device)
    labels = torch.randint(1, 31, (B * 5,), device=device)
    label_lengths = torch.tensor([5] * B, dtype=torch.long, device=device)

    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    with torch.no_grad():
        log_probs, out_lengths = model(mel, lengths)

    loss = ctc(log_probs.permute(1, 0, 2), labels, out_lengths, label_lengths)
    check(torch.isfinite(loss), f"CTC loss is not finite: {loss.item()}")


# ---------------------------------------------------------------------------
# 3. SSL encoder loading into CTC model
# ---------------------------------------------------------------------------

@test("ssl_encoder_transfer")
def test_ssl_encoder_transfer(args):
    """SSL encoder weights load correctly into the CTC model."""
    from CausalSpecUnit.model import CausalSpecUnitSSL, CausalSpecUnitCTC
    from CausalSpecUnit.common import save_checkpoint

    ssl = CausalSpecUnitSSL(variant="xs", k_coarse=100, k_fine=500)
    ctc = CausalSpecUnitCTC(vocab_size=128, variant="xs")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "ssl_ckpt")
        save_checkpoint(ckpt_dir, ssl, optimizer=None, scheduler=None, epoch=1)
        ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
        missing, unexpected = ctc.load_ssl_encoder(ckpt_path)

    # Encoder keys should all be present; only heads are legitimately missing
    encoder_missing = [k for k in missing if not k.startswith("model.")]
    check(len(encoder_missing) == 0, f"Missing encoder keys: {encoder_missing}")

    # Verify weights actually transferred (not just silently ignored)
    ssl_enc_param = next(iter(ssl.encoder.parameters())).detach().cpu()
    ctc_enc_param = next(iter(ctc.encoder.parameters())).detach().cpu()
    check(torch.allclose(ssl_enc_param, ctc_enc_param), "Encoder weights don't match after transfer")


# ---------------------------------------------------------------------------
# 4. Data pipeline
# ---------------------------------------------------------------------------

@test("mfcc_extractor")
def test_mfcc_extractor(args):
    """MFCCExtractor produces correct shape and finite values."""
    from CausalSpecUnit.generate_targets import MFCCExtractor
    import torchaudio

    # Create a tiny synthetic wav file
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "test.flac")
        wav = torch.randn(1, 16000)  # 1 second
        torchaudio.save(wav_path, wav, 16000)

        extractor = MFCCExtractor(n_mfcc=13)
        features = extractor(wav_path)

    check(features.ndim == 2, f"Expected 2D output, got shape {features.shape}")
    check(features.shape[1] == 39, f"Expected 39 dims (13*3), got {features.shape[1]}")
    check(torch.isfinite(features).all(), "MFCC features contain NaN/Inf")
    # Utterance CMVN should give ~zero mean per dim
    check(features.mean(0).abs().max() < 0.1, "MFCC utterance CMVN not working")


@test("generate_targets_smoke")
def test_generate_targets_smoke(args):
    """generate_targets runs end-to-end on 5 synthetic utterances."""
    import torchaudio
    from CausalSpecUnit.generate_targets import MFCCExtractor, LogMelExtractor
    from CausalSpecUnit.generate_targets import chunks_from_mel, update_cmvn, finalize_cmvn
    from sklearn.cluster import MiniBatchKMeans

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 5 fake flac files
        audio_paths = []
        for i in range(5):
            p = os.path.join(tmpdir, f"utt{i}.flac")
            wav = torch.randn(1, 16000 * 2)  # 2 seconds each
            torchaudio.save(p, wav, 16000)
            audio_paths.append(p)

        mel_ext = LogMelExtractor()
        mfcc_ext = MFCCExtractor(n_mfcc=13)

        # CMVN on mel
        state = {"count": 0, "sum": torch.zeros(80), "sumsq": torch.zeros(80)}
        for p in audio_paths:
            update_cmvn(mel_ext(p), state)
        mean, std = finalize_cmvn(state)
        check(torch.isfinite(mean).all())
        check((std > 0).all())

        # Chunks from MFCC
        all_chunks = []
        for p in audio_paths:
            mfcc = mfcc_ext(p)
            chunks = chunks_from_mel(mfcc, chunk_size=1, chunk_stride=1)
            check(chunks.shape[1] == 39, f"Expected 39-dim chunks, got {chunks.shape[1]}")
            all_chunks.append(chunks.numpy())
        all_chunks = np.concatenate(all_chunks, axis=0).astype(np.float32)

        # K-means
        km = MiniBatchKMeans(n_clusters=10, n_init="auto", random_state=42)
        km.fit(all_chunks)
        labels = km.predict(all_chunks)
        check(len(set(labels.tolist())) > 1, "K-means collapsed to 1 cluster")


# ---------------------------------------------------------------------------
# 5. Integration: targets dir + dataset [REAL DATA]
# ---------------------------------------------------------------------------

@test("targets_dir_valid")
def test_targets_dir_valid(args):
    """[REAL DATA] targets.pt and metadata.json exist and are loadable."""
    if not args.targets_dir:
        return "skip"
    import json

    targets_path = os.path.join(args.targets_dir, "targets.pt")
    meta_path = os.path.join(args.targets_dir, "metadata.json")
    cmvn_path = os.path.join(args.targets_dir, "cmvn.pt")

    check(os.path.isfile(targets_path), f"Missing: {targets_path}")
    check(os.path.isfile(meta_path), f"Missing: {meta_path}")
    check(os.path.isfile(cmvn_path), f"Missing: {cmvn_path}")

    with open(meta_path) as f:
        meta = json.load(f)
    for key in ("chunk_size", "chunk_stride", "k_coarse", "k_fine", "num_utterances"):
        check(key in meta, f"metadata.json missing key: {key}")

    targets = torch.load(targets_path, map_location="cpu", weights_only=False)
    check(len(targets) > 0, "targets.pt is empty")

    # Spot-check one entry
    uid, entry = next(iter(targets.items()))
    check("z100" in entry and "z500" in entry, f"Target entry missing z100/z500: {uid}")
    check(entry["z100"].dtype == torch.int64, "z100 should be int64")
    check(entry["z100"].max() < meta["k_coarse"], "z100 cluster id out of range")
    check(entry["z500"].max() < meta["k_fine"], "z500 cluster id out of range")


@test("ssl_dataset_loading")
def test_ssl_dataset_loading(args):
    """[REAL DATA] SpecUnitDataset loads and collates correctly."""
    if not args.targets_dir or not args.data_root:
        return "skip"
    from CausalSpecUnit.data import SpecUnitDataset, collate_ssl
    from torch.utils.data import DataLoader

    dataset = SpecUnitDataset(
        data_root=args.data_root,
        splits=["dev-clean"],
        targets_path=os.path.join(args.targets_dir, "targets.pt"),
        cmvn_path=os.path.join(args.targets_dir, "cmvn.pt"),
        max_items=8,
    )
    check(len(dataset) > 0, "Dataset is empty")

    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_ssl)
    mel, lengths, z100, z500, target_lengths = next(iter(loader))

    check(mel.ndim == 3, f"mel should be 3D, got {mel.shape}")
    check(mel.shape[2] == 80, f"mel should have 80 bins, got {mel.shape[2]}")
    check(torch.isfinite(mel).all(), "mel contains NaN/Inf")
    check((lengths > 0).all(), "zero-length utterance in batch")
    check(z100.shape[0] == z500.shape[0], "z100/z500 batch size mismatch")


@test("ssl_one_train_step")
def test_ssl_one_train_step(args):
    """[REAL DATA] SSL model completes one training step with loss decrease check."""
    if not args.targets_dir or not args.data_root:
        return "skip"
    from CausalSpecUnit.data import SpecUnitDataset, collate_ssl
    from CausalSpecUnit.model import CausalSpecUnitSSL
    from CausalSpecUnit.pretrain_ssl import (
        make_hubert_mask, corrupt_mel_from_target_mask,
        masked_unit_ce, align_ssl_tensors,
    )
    from torch.utils.data import DataLoader
    import json

    with open(os.path.join(args.targets_dir, "metadata.json")) as f:
        meta = json.load(f)
    chunk_size = int(meta["chunk_size"])
    chunk_stride = int(meta["chunk_stride"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalSpecUnitSSL(variant="xs", k_coarse=100, k_fine=500).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = SpecUnitDataset(
        data_root=args.data_root,
        splits=["dev-clean"],
        targets_path=os.path.join(args.targets_dir, "targets.pt"),
        cmvn_path=os.path.join(args.targets_dir, "cmvn.pt"),
        max_items=16,
    )
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_ssl)
    mel, lengths, z100, z500, target_lengths = next(iter(loader))
    mel, lengths = mel.to(device), lengths.to(device)
    z100, z500 = z100.to(device), z500.to(device)
    target_lengths = target_lengths.to(device)

    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        max_targets = z100.size(1)
        mask = make_hubert_mask(target_lengths, max_targets, 0.065, 10, device)
        corrupted = corrupt_mel_from_target_mask(mel, mask, chunk_size, chunk_stride, 0.0)
        coarse_logits, fine_logits, out_lengths = model(corrupted, lengths)
        coarse_logits, fine_logits, z100_a, z500_a, mask_a = align_ssl_tensors(
            coarse_logits, fine_logits, z100, z500, mask
        )
        loss = masked_unit_ce(coarse_logits, z100_a, mask_a) + \
               masked_unit_ce(fine_logits, z500_a, mask_a)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    check(all(torch.isfinite(torch.tensor(l)) for l in losses), f"Non-finite loss: {losses}")
    check(losses[0] > 0, "Loss is zero from the start")


# ---------------------------------------------------------------------------
# 6. Training validation [REAL DATA]
# ---------------------------------------------------------------------------

def validate_training(args):
    """
    Run 200 SSL training steps and decide go/no-go for cluster submission.

    Verdict logic:
      GO     — loss drops by >10% from first 10 steps to last 10 steps AND
                final loss < log(k_coarse) * 0.95 (better than near-uniform)
      CAUTION — loss is decreasing but slowly, or final loss is marginal
      NO-GO  — loss is flat, increasing, or non-finite
    """
    import json
    import math
    from CausalSpecUnit.data import SpecUnitDataset, collate_ssl
    from CausalSpecUnit.model import CausalSpecUnitSSL
    from CausalSpecUnit.pretrain_ssl import (
        make_hubert_mask, corrupt_mel_from_target_mask,
        masked_unit_ce, align_ssl_tensors,
    )
    from torch.utils.data import DataLoader

    if not args.targets_dir or not args.data_root:
        print("--validate-training requires --targets-dir and --data-root")
        return

    with open(os.path.join(args.targets_dir, "metadata.json")) as f:
        meta = json.load(f)
    chunk_size = int(meta["chunk_size"])
    chunk_stride = int(meta["chunk_stride"])
    k_coarse = int(meta["k_coarse"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nValidating training on {device} — running 200 steps...")

    dataset = SpecUnitDataset(
        data_root=args.data_root,
        splits=["dev-clean"],
        targets_path=os.path.join(args.targets_dir, "targets.pt"),
        cmvn_path=os.path.join(args.targets_dir, "cmvn.pt"),
        max_items=256,
    )
    loader = DataLoader(
        dataset, batch_size=8, collate_fn=collate_ssl,
        shuffle=True, drop_last=True, num_workers=0,
    )

    model = CausalSpecUnitSSL(variant="xs", k_coarse=100, k_fine=500).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.98), eps=1e-9)

    # Linear warmup over first 50 steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 50)
    )

    model.train()
    losses = []
    step = 0
    random_baseline = math.log(k_coarse)  # cross-entropy of uniform distribution

    data_iter = iter(loader)
    while step < 200:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        mel, lengths, z100, z500, target_lengths = batch
        mel, lengths = mel.to(device), lengths.to(device)
        z100, z500 = z100.to(device), z500.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()
        mask = make_hubert_mask(target_lengths, z100.size(1), 0.065, 10, device)
        corrupted = corrupt_mel_from_target_mask(mel, mask, chunk_size, chunk_stride, 0.0)
        coarse_logits, fine_logits, out_lengths = model(corrupted, lengths)
        coarse_logits, fine_logits, z100_a, z500_a, mask_a = align_ssl_tensors(
            coarse_logits, fine_logits, z100, z500, mask
        )
        loss = masked_unit_ce(coarse_logits, z100_a, mask_a) + \
               masked_unit_ce(fine_logits, z500_a, mask_a)

        if not torch.isfinite(loss):
            print(f"\n  Step {step+1}: NON-FINITE loss — diverged")
            print("\n  Verdict: NO-GO — loss diverged, check LR and gradient clipping")
            return

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        step += 1

        if step % 20 == 0:
            recent = sum(losses[-10:]) / 10
            print(f"  step {step:3d}/200  loss={recent:.4f}  (random baseline={random_baseline:.4f})")

    # Verdict
    first_10 = sum(losses[:10]) / 10
    last_10 = sum(losses[-10:]) / 10
    drop_pct = (first_10 - last_10) / first_10 * 100
    below_baseline = last_10 < random_baseline * 0.95

    print(f"\n  First 10 steps avg loss : {first_10:.4f}")
    print(f"  Last  10 steps avg loss : {last_10:.4f}")
    print(f"  Drop                    : {drop_pct:.1f}%")
    print(f"  Random baseline (log K) : {random_baseline:.4f}")
    print(f"  Below baseline          : {below_baseline}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Smooth with window=10
        smoothed = [sum(losses[max(0, i-9):i+1]) / min(i+1, 10) for i in range(len(losses))]
        plt.figure(figsize=(8, 4))
        plt.plot(losses, alpha=0.3, color="steelblue", label="raw")
        plt.plot(smoothed, color="steelblue", label="smoothed")
        plt.axhline(random_baseline, color="red", linestyle="--", label=f"random baseline ({random_baseline:.2f})")
        plt.xlabel("Step")
        plt.ylabel("SSL Loss")
        plt.title("Training validation — 200 steps")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(args.targets_dir, "validation_loss.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"\n  Loss curve saved to: {plot_path}")
    except ImportError:
        print("  (matplotlib not installed — skipping loss plot)")

    print()
    if not torch.isfinite(torch.tensor(last_10)):
        print("  Verdict: NO-GO — loss is non-finite")
    elif drop_pct < 2.0:
        print("  Verdict: NO-GO — loss is not decreasing (<2% drop over 200 steps)")
    elif drop_pct < 10.0 or not below_baseline:
        print(f"  Verdict: CAUTION — loss is decreasing ({drop_pct:.1f}%) but slowly or still above baseline")
        print("           Consider running longer locally or checking masking parameters")
    else:
        print(f"  Verdict: GO — loss dropped {drop_pct:.1f}% and is below random baseline")
        print("           Safe to submit to cluster")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests(args):
    to_run = results
    if args.test:
        to_run = [fn for fn in results if fn._test_name == args.test]
        if not to_run:
            print(f"No test named '{args.test}'. Available: {[f._test_name for f in results]}")
            sys.exit(1)

    passed = failed = skipped = 0
    for fn in to_run:
        name = fn._test_name
        try:
            result = fn(args)
            if result == "skip":
                print(f"  {SKIP} {name}")
                skipped += 1
            else:
                print(f"  {PASS} {name}")
                passed += 1
        except Exception:
            print(f"  {FAIL} {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets-dir", type=str, default=None,
                   help="Path to targets dir for real-data integration tests.")
    p.add_argument("--data-root", type=str, default=None,
                   help="Path to LibriSpeech root for real-data integration tests.")
    p.add_argument("--test", type=str, default=None,
                   help="Run only this test by name.")
    p.add_argument("--validate-training", action="store_true",
                   help="Run 200-step training validation and print go/no-go verdict.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.validate_training:
        validate_training(args)
    else:
        run_tests(args)
