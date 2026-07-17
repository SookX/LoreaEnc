"""Diagnose whether a distilled student actually learned frame-level teacher
structure, or merely collapsed onto a near-constant prediction.

Motivation
----------
Transformer features (HuBERT/wav2vec2) are strongly anisotropic: all frames
share a dominant mean direction, so a student that simply emits the *global
mean vector* can post a deceptively low cosine distance while carrying almost
no frame-level information. A distill run whose loss plateaus early and whose
encoder barely beats random init downstream is exactly what that looks like.

This script compares the student's cosine distance against trivial baselines on
the SAME frames, so the number becomes interpretable:

  student      1 - cos(student_pred, teacher)          <- what training reported
  mean-pred    1 - cos(global_mean_teacher, teacher)   <- predict a constant
  shuffled     1 - cos(teacher[perm], teacher)         <- a random OTHER frame

Read it as:
  student ~= mean-pred   -> COLLAPSE. The student learned ~nothing beyond the
                            mean; the checkpoint is not a useful encoder.
  student << mean-pred   -> the student learned real frame-level structure;
                            look downstream (fine-tuning) for the problem.
  shuffled  low (~0.2)   -> the teacher space is highly anisotropic, which is
                            why the raw cosine number looked "good".

Also reports a collapse ratio: the student's per-dim temporal std divided by
the teacher's. Near 0 means the student output barely moves over time.

Usage:
    python scripts/diagnose_distill.py \
        --checkpoint outputs/causal_specunit/distill_hubert_base_960h/checkpoint_step250000/checkpoint.pt
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from CausalSpecUnit.data import DistillDataset, collate_distill
from CausalSpecUnit.distill_pretrain import DistillStudent
from CausalSpecUnit.teacher import build_teacher


def cos_dist(a, b):
    return float((1.0 - F.cosine_similarity(a, b, dim=-1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="dataset/datasets/librispeech/LibriSpeech")
    ap.add_argument("--cmvn-path", default="outputs/causal_specunit/targets_960h_c8/cmvn.pt")
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--teacher", default="hubert_base")
    ap.add_argument("--teacher-layers", type=int, nargs="+", default=[3, 7, 11])
    ap.add_argument("--variant", default="xs")
    ap.add_argument("--num-utts", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  teacher={args.teacher}  layers={args.teacher_layers}")

    teacher = build_teacher(args.teacher, tuple(args.teacher_layers), downsample=True).to(device).eval()
    student = DistillStudent(args.variant, teacher.output_dim, len(args.teacher_layers)).to(device).eval()

    state = torch.load(args.checkpoint, map_location="cpu")
    model_state = state["model"] if "model" in state else state
    model_state = {
        k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in model_state.items()
    }
    missing, unexpected = student.load_state_dict(model_state, strict=False)
    print(f"student load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  !! missing keys (first 5): {list(missing)[:5]}", file=sys.stderr)
    step = state.get("optimizer_steps", "?")
    print(f"checkpoint optimizer_steps={step}")

    ds = DistillDataset(args.data_root, [args.split], args.cmvn_path, max_items=args.num_utts)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_distill)

    n_layers = len(args.teacher_layers)
    stu_frames = [[] for _ in range(n_layers)]
    tea_frames = [[] for _ in range(n_layers)]

    with torch.no_grad():
        for mel, lengths, wavs, wav_lengths in loader:
            mel, lengths = mel.to(device), lengths.to(device)
            wavs, wav_lengths = wavs.to(device), wav_lengths.to(device)
            t_feats, t_len = teacher(wavs, wav_lengths)
            preds, s_len = student(mel, lengths)
            for i, (p, t) in enumerate(zip(preds, t_feats)):
                T = min(p.size(1), t.size(1))
                p, t = p[:, :T].float(), t[:, :T].float()
                valid = torch.minimum(s_len, t_len).clamp(max=T)
                mask = torch.arange(T, device=device)[None, :] < valid[:, None]
                sel = mask.unsqueeze(-1).expand_as(p)
                stu_frames[i].append(p[sel].view(-1, p.size(-1)).cpu())
                tea_frames[i].append(t[sel].view(-1, t.size(-1)).cpu())

    print()
    print("=" * 78)
    print(f"Distillation diagnosis on {args.split} ({args.num_utts} utts)")
    print("=" * 78)
    header = f"{'layer':>5} {'frames':>8} {'student':>9} {'mean-pred':>10} {'shuffled':>9} {'std ratio':>10}"
    print(header)
    print("-" * len(header))

    verdicts = []
    for i, layer in enumerate(args.teacher_layers):
        S = torch.cat(stu_frames[i])
        Tt = torch.cat(tea_frames[i])
        student_cd = cos_dist(S, Tt)
        mu = Tt.mean(dim=0, keepdim=True).expand_as(Tt)
        mean_cd = cos_dist(mu, Tt)
        perm = torch.randperm(Tt.size(0))
        shuf_cd = cos_dist(Tt[perm], Tt)
        std_ratio = float(S.std(dim=0).mean() / Tt.std(dim=0).mean())
        print(f"{layer:>5} {S.size(0):>8} {student_cd:>9.4f} {mean_cd:>10.4f} {shuf_cd:>9.4f} {std_ratio:>10.4f}")
        verdicts.append((layer, student_cd, mean_cd, std_ratio))

    print()
    print("=" * 78)
    print("Verdict")
    print("=" * 78)
    for layer, s, m, r in verdicts:
        # How much of the gap between "predict the mean" and "perfect" did the
        # student actually close? mean_cd is the trivial floor.
        closed = (m - s) / m * 100 if m > 1e-8 else 0.0
        if s > m * 0.9:
            tag = "COLLAPSED - no better than predicting the constant mean"
        elif closed < 40:
            tag = f"WEAK - closed only {closed:.0f}% of the gap vs the mean-predictor"
        else:
            tag = f"OK - closed {closed:.0f}% of the gap vs the mean-predictor"
        extra = "  [output nearly constant over time]" if r < 0.25 else ""
        print(f"  layer {layer}: {tag}{extra}")
    print()


if __name__ == "__main__":
    main()
