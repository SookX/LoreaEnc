"""
SqueezeFormer XS training script.

Connects to dataset/dataset.py (LibriSpeechDataset, mel mode) and trains
SqueezeFormer XS with CTC loss, matching the original paper setup.

Usage:
    python SqueezeFormer/train.py \
        --data_root /path/to/LibriSpeech \
        --manifest_path /path/to/manifest.json \   # optional
        --epochs 100 \
        --batch_size 32 \
        --variant xs
"""

import sys
import os
import math
import random
import argparse
import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2CTCTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset.dataset import LibriSpeechDataset
from SqueezeFormer import Squeezeformer, get_config


# ─── Data ────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Pad mel features and labels to max length in batch."""
    mel_list    = [item["input_values"] for item in batch]   # each: [T, 80]
    label_list  = [item["labels"]       for item in batch]
    lengths     = torch.tensor([m.size(0) for m in mel_list], dtype=torch.long)

    mel_padded  = nn.utils.rnn.pad_sequence(mel_list, batch_first=True)  # [B, T_max, 80]

    label_lengths = torch.tensor([len(l) for l in label_list], dtype=torch.long)
    labels_flat   = torch.cat([
        l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in label_list
    ])

    return mel_padded, lengths, labels_flat, label_lengths


# ─── Model ───────────────────────────────────────────────────────────────────

def build_model(config, num_classes: int) -> Squeezeformer:
    return Squeezeformer(
        num_classes=num_classes,
        input_dim=config.input_dim,
        encoder_dim=config.encoder_dim,
        num_encoder_layers=config.num_encoder_layers,
        reduce_layer_index=config.reduce_layer_index,
        recover_layer_index=config.recover_layer_index,
        num_attention_heads=config.num_attention_heads,
        feed_forward_expansion_factor=config.feed_forward_expansion_factor,
        conv_expansion_factor=config.conv_expansion_factor,
        input_dropout_p=config.input_dropout_p,
        feed_forward_dropout_p=config.feed_forward_dropout_p,
        attention_dropout_p=config.attention_dropout_p,
        conv_dropout_p=config.conv_dropout_p,
        conv_kernel_size=config.conv_kernel_size,
        half_step_residual=config.half_step_residual,
        adaptive_scale=config.adaptive_scale,
    )


# ─── LR schedule ─────────────────────────────────────────────────────────────

def warmup_cosine_lr(optimizer, step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        lr = base_lr * max(step, 1) / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ─── Greedy CTC decode (for live WER estimate) ───────────────────────────────

def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int):
    """Collapse greedy argmax sequence, remove blanks and repeats."""
    preds = log_probs.argmax(dim=-1)   # [B, T]
    decoded = []
    for b in range(preds.size(0)):
        seq, prev = [], None
        for tok in preds[b, :lengths[b]].tolist():
            if tok != prev:
                seq.append(tok)
            prev = tok
        decoded.append([t for t in seq if t != blank_id])
    return decoded


def _edit_distance(a, b):
    """Wagner-Fischer edit distance (stdlib only)."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def word_error_rate(hyp_text: str, ref_text: str) -> float:
    """WER = edit distance at word level / number of reference words."""
    hyp_words = hyp_text.lower().split()
    ref_words = ref_text.lower().split()
    return _edit_distance(hyp_words, ref_words) / max(len(ref_words), 1)


def token_error_rate(hyps, refs):
    """Token-level edit distance averaged over batch."""
    total_err, total_ref = 0, 0
    for h, r in zip(hyps, refs):
        total_err += _edit_distance(h, r)
        total_ref += max(len(r), 1)
    return total_err / max(total_ref, 1)


# ─── Training ────────────────────────────────────────────────────────────────

def evaluate_sample(model, dev_dataset, tokenizer, blank_id, device):
    """Pick a random dev-clean sample, decode it, and return (prediction, reference, WER)."""
    model.eval()
    idx = random.randrange(len(dev_dataset))
    sample = dev_dataset[idx]

    mel = sample["input_values"].unsqueeze(0).to(device)        # [1, T, 80]
    length = torch.tensor([mel.size(1)], dtype=torch.long).to(device)
    ref_text = sample["raw_transcript"]

    with torch.no_grad():
        log_probs, out_lengths = model(mel, length)
        hyp_tokens = greedy_decode(log_probs, out_lengths, blank_id)[0]

    hyp_text = tokenizer.decode(hyp_tokens)
    wer = word_error_rate(hyp_text, ref_text)
    model.train()
    return hyp_text, ref_text, wer


def main():
    parser = argparse.ArgumentParser(description="Train SqueezeFormer on LibriSpeech")
    parser.add_argument("--data_root",      type=str, required=True)
    parser.add_argument("--manifest_path",  type=str, default=None)
    parser.add_argument("--splits",         nargs="+", default=["train-clean-100"])
    parser.add_argument("--variant",        type=str, default="xs")
    parser.add_argument("--epochs",         type=int, default=100)
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--warmup_epochs",  type=int, default=10)
    parser.add_argument("--save_dir",       type=str, default="./work_dir/squeezeformer")
    parser.add_argument("--save_every",     type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--num_workers",    type=int, default=0)
    parser.add_argument("--resume",         type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_ter = True

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer  = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
    blank_id   = tokenizer.pad_token_id
    num_classes = tokenizer.vocab_size

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = LibriSpeechDataset(
        path_to_data_root=args.data_root,
        include_splits=args.splits if args.manifest_path is None else None,
        manifest_path=args.manifest_path,
        tokenizer=tokenizer,
        train_split=True,
        apply_spec_augment=True,
        mode="mel",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    dev_dataset = LibriSpeechDataset(
        path_to_data_root=args.data_root,
        include_splits=["dev-clean"],
        tokenizer=tokenizer,
        train_split=False,
        apply_spec_augment=False,
        mode="mel",
    )

    # ── Model ──────────────────────────────────────────────────────────────
    config = get_config(args.variant)
    model  = build_model(config, num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6, betas=(0.9, 0.98))
    ctc_loss  = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    total_steps  = args.epochs * len(loader)
    warmup_steps = args.warmup_epochs * len(loader)
    global_step  = (start_epoch - 1) * len(loader)

    # ── Epoch progress bar (outer) ─────────────────────────────────────────
    n_params = model.count_parameters()
    epoch_bar = tqdm(
        range(start_epoch, args.epochs + 1),
        desc=f"SqueezeFormer-{args.variant.upper()} ({n_params/1e6:.1f}M) | {device}",
        unit="ep",
        dynamic_ncols=True,
        position=0,
    )

    # Rolling window for smoothed loss (last 50 steps)
    loss_window = collections.deque(maxlen=50)
    epoch_history = []   # (epoch, avg_loss, avg_ter)

    for epoch in epoch_bar:
        model.train()
        epoch_loss   = 0.0
        epoch_ter    = 0.0
        ter_batches  = 0

        # ── Batch progress bar (inner) ────────────────────────────────────
        batch_bar = tqdm(
            loader,
            desc=f"  Epoch {epoch:03d}",
            unit="batch",
            dynamic_ncols=True,
            position=1,
            leave=False,
        )

        for batch in batch_bar:
            mel, lengths, labels, label_lengths = [b.to(device) for b in batch]

            lr = warmup_cosine_lr(optimizer, global_step, warmup_steps, total_steps, args.lr)
            optimizer.zero_grad()

            log_probs, output_lengths = model(mel, lengths)
            log_probs_ctc = log_probs.permute(1, 0, 2)   # (T, B, C)
            loss = ctc_loss(log_probs_ctc, labels, output_lengths, label_lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            loss_val = loss.item()
            loss_window.append(loss_val)
            epoch_loss += loss_val
            global_step += 1

            # Live TER estimate (only if editdistance is available)
            ter_str = ""
            if use_ter:
                with torch.no_grad():
                    hyps = greedy_decode(log_probs.detach(), output_lengths, blank_id)
                # Rebuild per-sample reference token lists
                offset = 0
                refs = []
                for ll in label_lengths.tolist():
                    refs.append(labels[offset:offset + ll].tolist())
                    offset += ll
                ter = token_error_rate(hyps, refs)
                epoch_ter   += ter
                ter_batches += 1
                ter_str = f"  TER {ter:.2%}"

            smoothed = sum(loss_window) / len(loss_window)
            batch_bar.set_postfix_str(
                f"loss {loss_val:.3f}  smooth {smoothed:.3f}{ter_str}  lr {lr:.1e}",
                refresh=False,
            )

        # ── End-of-epoch summary ──────────────────────────────────────────
        avg_loss = epoch_loss / len(loader)
        avg_ter  = (epoch_ter / ter_batches) if ter_batches > 0 else float("nan")
        epoch_history.append((epoch, avg_loss, avg_ter))

        # ── Dev-clean sample prediction ───────────────────────────────────
        hyp, ref, wer = evaluate_sample(model, dev_dataset, tokenizer, blank_id, device)
        tqdm.write(
            f"\n  [epoch {epoch:03d}]"
            f"\n    REF : {ref}"
            f"\n    HYP : {hyp}"
            f"\n    WER : {wer:.2%}\n"
        )

        ter_summary = f"  avg-TER {avg_ter:.2%}" if use_ter else ""
        epoch_bar.set_postfix_str(
            f"loss {avg_loss:.4f}{ter_summary}  WER {wer:.2%}  lr {lr:.1e}",
            refresh=True,
        )

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"squeezeformer_{args.variant}_ep{epoch:03d}.pt")
            torch.save({
                "epoch":   epoch,
                "model":   model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": epoch_history,
            }, ckpt_path)
            tqdm.write(f"  [ckpt] saved → {ckpt_path}")


if __name__ == "__main__":
    main()
