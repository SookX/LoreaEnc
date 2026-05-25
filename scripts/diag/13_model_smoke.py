"""Smoke test the MelHuBERT-Transformer CTC pipeline on a SINGLE GPU
with synthetic data and no dataloader.

If this finishes, the model + SSL checkpoint load + autograd graph + the
CTC head are all working. Any hang in the real training run is then due
to (a) DDP / NCCL, or (b) the dataloader, not the model.

Run via Slurm:
    sbatch scripts/diag/13_model_smoke.sh

Or directly on a node with 1 GPU:
    python scripts/diag/13_model_smoke.py \
        --ssl-checkpoint outputs/causal_specunit/melhubert_transformer_mh9m/ssl_fine_150000/checkpoint_step150000/checkpoint.pt
"""

import argparse
import time

import torch
import torch.nn as nn

from CausalSpecUnit.model import CausalSpecUnitCTC


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ssl-checkpoint", type=str, required=True)
    p.add_argument("--variant", type=str, default="mh9m")
    p.add_argument("--vocab-size", type=int, default=128 + 1)  # bpe128 + blank
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=1600)  # ~16 s @ 100 Hz
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--label-len", type=int, default=80)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--use-bf16", action="store_true", default=True)
    args = p.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    log(f"building CausalSpecUnitCTC variant={args.variant} vocab={args.vocab_size}")
    model = CausalSpecUnitCTC(vocab_size=args.vocab_size, variant=args.variant)
    log(f"  encoder.is_transformer_baseline={getattr(model, 'is_transformer_baseline', None)}")

    log(f"loading SSL checkpoint: {args.ssl_checkpoint}")
    missing, unexpected = model.load_ssl_encoder(args.ssl_checkpoint, map_location="cpu", load_ssl_heads=False)
    log(f"  missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        log(f"  missing (first 5): {missing[:5]}")
    if unexpected[:5]:
        log(f"  unexpected (first 5): {unexpected[:5]}")

    model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  model on cuda. params={n_params/1e6:.2f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    blank_id = 0
    ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    B, T, M = args.batch_size, args.seq_len, args.n_mels
    L = args.label_len

    log(f"synthetic batch: mel={B}x{T}x{M}, label_len={L}")

    for step in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.time()

        mel = torch.randn(B, T, M, device=device)
        lengths = torch.full((B,), T, dtype=torch.long, device=device)
        labels = torch.randint(1, args.vocab_size, (B, L), device=device, dtype=torch.long)
        label_lengths = torch.full((B,), L, dtype=torch.long, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_bf16):
            log_probs, output_lengths, _, _ = model(mel, lengths, return_inter=False, return_ssl=False)
            loss = ctc_loss(log_probs.transpose(0, 1), labels, output_lengths, label_lengths)

        t_fwd = time.time() - t0
        torch.cuda.synchronize()
        t1 = time.time()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = time.time() - t1
        t2 = time.time()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        torch.cuda.synchronize()
        t_step = time.time() - t2

        log(
            f"step {step:02d} loss={loss.item():.3f} grad_norm={grad_norm.item():.3f} "
            f"fwd={t_fwd*1000:.0f}ms bwd={t_bwd*1000:.0f}ms step={t_step*1000:.0f}ms"
        )

    log("smoke test PASSED")


if __name__ == "__main__":
    main()
