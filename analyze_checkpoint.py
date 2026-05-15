"""Probe a trained SSL checkpoint to see whether the encoder learned useful structure.

Runs the checkpoint on real dev-clean data and reports:
  - prediction accuracy at masked positions vs unmasked positions
  - entropy of predicted distributions (low = confident, high = random)
  - cluster usage in predictions (how many of the K classes does the model ever pick?)
  - per-position feature variance (do features differ across time?)
"""
import argparse
import math
import os

import torch
import torch.nn.functional as F

from CausalSpecUnit.data import SpecUnitDataset, collate_ssl
from CausalSpecUnit.model import CausalSpecUnitSSL
from CausalSpecUnit.pretrain_ssl import (
    align_ssl_tensors,
    corrupt_mel_from_target_mask,
    make_hubert_mask,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="dataset")
    p.add_argument("--splits", nargs="+", default=["dev-clean"])
    p.add_argument("--targets-dir", type=str, default="outputs/causal_specunit/targets_debug_c2")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--mask-prob", type=float, default=0.08)
    p.add_argument("--mask-length", type=int, default=10)
    p.add_argument("--chunk-size", type=int, default=2)
    p.add_argument("--chunk-stride", type=int, default=4)
    p.add_argument("--variant", type=str, default="xs")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  steps={ckpt.get('optimizer_steps')} epoch={ckpt.get('epoch')} ssl_loss={ckpt.get('ssl_loss'):.3f}")

    print(f"\nBuilding model ({args.variant}) and loading weights...")
    model = CausalSpecUnitSSL(variant=args.variant).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    # mask_emb learned change
    init = torch.empty_like(model.mask_emb).normal_(mean=0.0, std=0.1)
    me_norm = model.mask_emb.norm().item()
    me_diff = (model.mask_emb - init).norm().item()
    print(f"\n[mask_emb] norm={me_norm:.3f} (init~0.9 expected) "
          f"changed_from_init=~{me_diff:.3f} (large means learned)")

    # ---- load data ----
    print(f"\nLoading {args.num_samples} samples from {args.splits} ...")
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

    # ---- prediction accuracy WITHOUT masking (probe feature quality directly) ----
    print(f"\n[A] Clean-input prediction accuracy (no masking, all positions)")
    coarse_logits, fine_logits, _ = model(mel, lengths)
    cl, fl, z1, z5, _ = align_ssl_tensors(
        coarse_logits, fine_logits, z100, z500,
        torch.zeros_like(z100, dtype=torch.bool),
    )
    valid = z1.ne(-100)
    pred_c = cl.argmax(dim=-1)
    pred_f = fl.argmax(dim=-1)
    acc_c = (pred_c[valid] == z1[valid]).float().mean().item()
    acc_f = (pred_f[valid] == z5[valid]).float().mean().item()
    top5_c = (cl.topk(5, dim=-1).indices == z1.unsqueeze(-1))[valid].any(-1).float().mean().item()
    top5_f = (fl.topk(5, dim=-1).indices == z5.unsqueeze(-1))[valid].any(-1).float().mean().item()
    print(f"  coarse (K=100): top1={acc_c:.2%}  top5={top5_c:.2%}  random_baseline={1/100:.2%}")
    print(f"  fine   (K=500): top1={acc_f:.2%}  top5={top5_f:.2%}  random_baseline={1/500:.2%}")
    print(f"  -> ratio over random: coarse={acc_c*100:.1f}x  fine={acc_f*500:.1f}x")

    # ---- prediction accuracy AT masked positions (the actual SSL task) ----
    print(f"\n[B] Masked-position prediction accuracy (the SSL objective)")
    m = make_hubert_mask(target_lengths, z100.size(1), args.mask_prob, args.mask_length, device)
    mt = getattr(model, "mask_emb", None)
    corrupted = corrupt_mel_from_target_mask(mel, m, args.chunk_size, args.chunk_stride, mt if mt is not None else 0.0)
    cl, fl, _ = model(corrupted, lengths)
    cl, fl, z1, z5, m_aligned = align_ssl_tensors(cl, fl, z100, z500, m)
    valid = m_aligned & z1.ne(-100)
    n_masked = valid.sum().item()
    if n_masked > 0:
        pred_c = cl.argmax(dim=-1)
        pred_f = fl.argmax(dim=-1)
        acc_c = (pred_c[valid] == z1[valid]).float().mean().item()
        acc_f = (pred_f[valid] == z5[valid]).float().mean().item()
        top5_c = (cl.topk(5, dim=-1).indices == z1.unsqueeze(-1))[valid].any(-1).float().mean().item()
        top5_f = (fl.topk(5, dim=-1).indices == z5.unsqueeze(-1))[valid].any(-1).float().mean().item()
        loss_c = F.cross_entropy(cl[valid], z1[valid]).item()
        loss_f = F.cross_entropy(fl[valid], z5[valid]).item()
        print(f"  N masked positions: {n_masked}")
        print(f"  coarse (K=100): top1={acc_c:.2%}  top5={top5_c:.2%}  loss={loss_c:.3f}  random_loss={math.log(100):.3f}")
        print(f"  fine   (K=500): top1={acc_f:.2%}  top5={top5_f:.2%}  loss={loss_f:.3f}  random_loss={math.log(500):.3f}")
        print(f"  -> ratio over random: coarse={acc_c*100:.1f}x  fine={acc_f*500:.1f}x")

    # ---- prediction entropy (sharpness of distribution) ----
    print(f"\n[C] Prediction confidence (entropy)")
    pc = F.softmax(cl[valid], dim=-1)
    pf = F.softmax(fl[valid], dim=-1)
    ent_c = -(pc * pc.clamp_min(1e-12).log()).sum(-1).mean().item()
    ent_f = -(pf * pf.clamp_min(1e-12).log()).sum(-1).mean().item()
    print(f"  coarse entropy: {ent_c:.3f} (uniform={math.log(100):.3f}, perfect=0)")
    print(f"  fine   entropy: {ent_f:.3f} (uniform={math.log(500):.3f}, perfect=0)")
    print(f"  -> coarse confidence ratio: {1 - ent_c / math.log(100):.1%} (higher = sharper)")
    print(f"  -> fine   confidence ratio: {1 - ent_f / math.log(500):.1%}")

    # ---- cluster usage diversity ----
    print(f"\n[D] Cluster usage diversity (does model use full vocabulary?)")
    pred_c_all = cl.argmax(dim=-1)
    pred_f_all = fl.argmax(dim=-1)
    used_c = pred_c_all.unique().numel()
    used_f = pred_f_all.unique().numel()
    print(f"  coarse: model picks {used_c}/100 distinct clusters")
    print(f"  fine:   model picks {used_f}/500 distinct clusters")
    if used_c < 20:
        print(f"  [WARN] model is collapsing to a few coarse clusters ({used_c}/100)")
    if used_f < 50:
        print(f"  [WARN] model is collapsing to a few fine clusters ({used_f}/500)")

    # ---- feature variance across time (are features positionally distinct?) ----
    print(f"\n[E] Encoder feature variance")
    encoder = model.encoder
    h, h_lengths = encoder(mel, lengths)
    # Mean across time, std across time
    feat_std = h.std(dim=1).mean().item()
    feat_mean = h.mean().item()
    print(f"  feature mean across all time/batch: {feat_mean:.4f}")
    print(f"  std of features across time (avg over feature dims & batch): {feat_std:.4f}")
    if feat_std < 0.01:
        print(f"  [WARN] features are nearly constant across time — encoder collapsed")

    print("\n" + "=" * 50)
    print("Diagnosis:")
    if acc_c > 5 * (1 / 100) and acc_f > 5 * (1 / 500):
        print("  features predict targets well above random -> SSL learned useful structure")
    elif acc_c > 2 * (1 / 100):
        print("  features predict above random but weakly -> partial SSL signal, might still help downstream")
    else:
        print("  features near random on the SSL task -> SSL did not learn meaningful structure")


if __name__ == "__main__":
    main()