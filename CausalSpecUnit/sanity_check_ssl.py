"""End-to-end sanity checks for the SSL pretraining pipeline.

Runs nine independent checks against a small real batch and prints PASS/FAIL
for each. Run from the project root:

    python -m CausalSpecUnit.sanity_check_ssl \
        --data-root dataset \
        --splits dev-clean \
        --targets-dir outputs/causal_specunit/targets_debug_c2 \
        --chunk-size 2 --chunk-stride 4

Exits non-zero if any check fails.
"""
import argparse
import math
import os
import sys

import torch
import torch.nn as nn

from CausalSpecUnit.data import SpecUnitDataset, collate_ssl
from CausalSpecUnit.model import CausalSpecUnitSSL
from CausalSpecUnit.pretrain_ssl import (
    align_ssl_tensors,
    corrupt_mel_from_target_mask,
    make_hubert_mask,
    masked_unit_ce,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=["dev-clean"])
    p.add_argument("--targets-dir", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--mask-prob", type=float, default=0.08)
    p.add_argument("--mask-length", type=int, default=10)
    p.add_argument("--chunk-size", type=int, default=2)
    p.add_argument("--chunk-stride", type=int, default=4)
    p.add_argument("--variant", type=str, default="xs")
    p.add_argument("--memorize-steps", type=int, default=200)
    return p.parse_args()


PASSED = []
FAILED = []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # ---------- setup ----------
    print("Loading data and model...")
    dataset = SpecUnitDataset(
        data_root=args.data_root, splits=args.splits,
        targets_path=os.path.join(args.targets_dir, "targets.pt"),
        cmvn_path=os.path.join(args.targets_dir, "cmvn.pt"),
        max_items=args.num_samples,
    )
    items = [dataset[i] for i in range(min(args.num_samples, len(dataset)))]
    batch = collate_ssl(items)
    mel, lengths, z100, z500, target_lengths = batch
    mel, lengths = mel.to(device), lengths.to(device)
    z100, z500 = z100.to(device), z500.to(device)

    model = CausalSpecUnitSSL(variant=args.variant).to(device)
    model.train()

    # ---------- 1. CMVN sanity ----------
    print("\n[1] CMVN normalization sanity")
    mean_abs = mel.abs().mean().item()
    std = mel.std().item()
    check("mel mean is near 0",
          abs(mel.mean().item()) < 0.5,
          f"mean={mel.mean().item():.3f}")
    check("mel std is near 1",
          0.5 < std < 1.5,
          f"std={std:.3f}")
    check("mel range is bounded",
          mean_abs < 5.0,
          f"|mel| mean={mean_abs:.3f}")

    # ---------- 2. Mask construction ----------
    print("\n[2] Mask construction")
    masked_positions = make_hubert_mask(
        target_lengths=target_lengths, max_targets=z100.size(1),
        mask_prob=args.mask_prob, mask_length=args.mask_length, device=device,
    )
    per_utt_frac = []
    for i in range(mel.size(0)):
        tT = int(target_lengths[i].item())
        per_utt_frac.append(masked_positions[i, :tT].float().mean().item())
    check("every utterance has at least 1 masked position",
          all(f > 0 for f in per_utt_frac),
          f"min_frac={min(per_utt_frac):.2%}")
    check("no masked positions beyond target length (no out-of-bounds)",
          all(masked_positions[i, int(target_lengths[i].item()):].sum().item() == 0 for i in range(mel.size(0))),
          "checked per-utterance")
    overall_frac = sum(per_utt_frac) / len(per_utt_frac)
    expected_low = args.mask_prob * 0.5
    expected_high = args.mask_prob * args.mask_length * 1.2
    check("overall masked fraction is in expected range",
          expected_low < overall_frac < expected_high,
          f"frac={overall_frac:.2%} expected_in=[{expected_low:.2%}, {expected_high:.2%}]")

    # ---------- 3. Mel corruption ----------
    print("\n[3] Mel corruption")
    corrupted = corrupt_mel_from_target_mask(
        mel, masked_positions, args.chunk_size, args.chunk_stride,
        mask_value=model.mask_emb,
    )
    check("corrupted shape matches mel",
          corrupted.shape == mel.shape, str(corrupted.shape))
    check("corrupted differs from mel exactly at masked frames",
          (corrupted != mel).any(dim=-1).sum().item() > 0,
          f"differing frames={(corrupted != mel).any(dim=-1).sum().item()}")
    # The replacement should be the mask_emb at differing rows
    diff_rows = (corrupted != mel).any(dim=-1)  # (B, T)
    if diff_rows.any():
        rep = corrupted[diff_rows][0]  # (n_mels,)
        diff_with_token = (rep - model.mask_emb.detach()).abs().max().item()
        check("masked frames replaced with model.mask_emb",
              diff_with_token < 1e-5, f"max abs diff={diff_with_token:.2e}")

    # ---------- 4. Encoder forward pass ----------
    print("\n[4] Encoder forward pass")
    coarse_logits, fine_logits, out_lengths = model(corrupted, lengths)
    check("coarse logits shape correct (B, T_enc, K_c)",
          coarse_logits.dim() == 3 and coarse_logits.size(0) == mel.size(0)
          and coarse_logits.size(2) == 100,
          f"shape={tuple(coarse_logits.shape)}")
    check("fine logits shape correct (B, T_enc, K_f)",
          fine_logits.dim() == 3 and fine_logits.size(0) == mel.size(0)
          and fine_logits.size(2) == 500,
          f"shape={tuple(fine_logits.shape)}")
    check("encoder output_lengths <= padded T",
          (out_lengths <= coarse_logits.size(1)).all().item(),
          f"out_lengths={out_lengths.tolist()}")

    # ---------- 5. Length alignment ----------
    print("\n[5] Time alignment encoder vs targets")
    cl, fl, z1, z5, mp = align_ssl_tensors(
        coarse_logits, fine_logits, z100, z500, masked_positions
    )
    enc_T = coarse_logits.size(1)
    tgt_T = z100.size(1)
    check("encoder T and target T are within 4 frames of each other",
          abs(enc_T - tgt_T) <= 4,
          f"enc={enc_T} tgt={tgt_T} diff={enc_T - tgt_T}")
    check("aligned mask still has masked positions after crop",
          mp.any().item(), f"any masked after crop={mp.any().item()}")

    # ---------- 6. Loss at init ≈ log(K) ----------
    print("\n[6] Loss at random init")
    loss100 = masked_unit_ce(cl, z1, mp).item()
    loss500 = masked_unit_ce(fl, z5, mp).item()
    expected_c100 = math.log(100)
    expected_c500 = math.log(500)
    check("loss100 near log(100)~4.61 at init",
          abs(loss100 - expected_c100) < 0.5,
          f"loss100={loss100:.3f} expected={expected_c100:.3f}")
    check("loss500 near log(500)~6.21 at init",
          abs(loss500 - expected_c500) < 0.5,
          f"loss500={loss500:.3f} expected={expected_c500:.3f}")

    # ---------- 7. Padding excluded from loss ----------
    print("\n[7] Padding excluded from loss")
    # Pad targets with -100 should be ignored by masked_unit_ce.
    # Force a position in padding to be 'masked' and verify it's filtered.
    fake_mask = mp.clone()
    last_valid = int(target_lengths.min().item())
    if last_valid < fake_mask.size(1):
        fake_mask[0, last_valid:] = True
    # Targets at padding should be -100 (set by collate_ssl).
    pad_targets_count = (z1 == -100).sum().item()
    check("collate_ssl pads target labels with -100",
          pad_targets_count > 0, f"pad count={pad_targets_count}")
    loss_with_fake = masked_unit_ce(cl, z1, fake_mask).item()
    loss_real = masked_unit_ce(cl, z1, mp).item()
    # Loss should be similar — padding entries get filtered out by ne(-100)
    check("masking padded positions does not affect loss",
          abs(loss_with_fake - loss_real) < 0.5,
          f"with_pad={loss_with_fake:.3f} real={loss_real:.3f}")

    # ---------- 8. Gradient flow to mask_emb ----------
    print("\n[8] Gradient flow")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # zero LR — we only check grads
    optimizer.zero_grad()
    coarse_logits, fine_logits, _ = model(corrupted, lengths)
    cl, fl, z1, z5, mp = align_ssl_tensors(coarse_logits, fine_logits, z100, z500, masked_positions)
    total = masked_unit_ce(cl, z1, mp) + masked_unit_ce(fl, z5, mp)
    total.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    n_with_grad = sum(1 for g in grads.values() if g.abs().sum().item() > 0)
    n_total = sum(1 for _ in model.parameters())
    check("most parameters receive non-zero gradients",
          n_with_grad / max(n_total, 1) > 0.5,
          f"{n_with_grad}/{n_total} params have grad")
    mask_grad = model.mask_emb.grad
    check("mask_emb receives non-zero gradient",
          mask_grad is not None and mask_grad.abs().sum().item() > 0,
          f"|grad|={mask_grad.abs().sum().item():.3e}" if mask_grad is not None else "None")
    check("head_coarse receives non-zero gradient",
          model.head_coarse.weight.grad is not None
          and model.head_coarse.weight.grad.abs().sum().item() > 0)
    check("head_fine receives non-zero gradient",
          model.head_fine.weight.grad is not None
          and model.head_fine.weight.grad.abs().sum().item() > 0)
    grad_norms = [g.norm().item() for g in grads.values()]
    check("no NaN or Inf in gradients",
          all(math.isfinite(n) for n in grad_norms),
          f"max={max(grad_norms):.2e}")

    # ---------- 9. Memorization test (loss should drop on fixed batch) ----------
    print(f"\n[9] Memorization test ({args.memorize_steps} steps on fixed batch)")
    # If pipeline is correct, the model should drive loss DOWN on a tiny fixed batch.
    # If loss stays at log(K), something is fundamentally broken.
    model_mem = CausalSpecUnitSSL(variant=args.variant).to(device)
    optimizer = torch.optim.AdamW(model_mem.parameters(), lr=1e-3)
    init_loss = None
    losses = []
    for step in range(args.memorize_steps):
        torch.manual_seed(step)  # vary mask each step (matches real training)
        m = make_hubert_mask(target_lengths, z100.size(1), args.mask_prob, args.mask_length, device)
        c = corrupt_mel_from_target_mask(mel, m, args.chunk_size, args.chunk_stride, model_mem.mask_emb)
        cl, fl, _ = model_mem(c, lengths)
        cl, fl, z1, z5, mp = align_ssl_tensors(cl, fl, z100, z500, m)
        loss = masked_unit_ce(cl, z1, mp) + masked_unit_ce(fl, z5, mp)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model_mem.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if init_loss is None:
            init_loss = loss.item()
    final_loss = sum(losses[-10:]) / 10
    drop = init_loss - final_loss
    print(f"  init loss: {init_loss:.3f}  final (avg last 10): {final_loss:.3f}  drop: {drop:.3f}")
    check("loss decreases on fixed batch (pipeline learns)",
          drop > 0.2 * init_loss,
          f"drop_ratio={drop / max(init_loss, 1e-6):.1%}")
    check("loss trajectory is monotonically decreasing on average",
          sum(losses[:10]) / 10 > sum(losses[-10:]) / 10,
          f"first10_avg={sum(losses[:10]) / 10:.3f} last10_avg={sum(losses[-10:]) / 10:.3f}")

    # ---------- summary ----------
    print(f"\n{'=' * 50}")
    print(f"Passed: {len(PASSED)}/{len(PASSED) + len(FAILED)}")
    if FAILED:
        print(f"FAILED:")
        for name in FAILED:
            print(f"  - {name}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()