"""
SqueezeFormer XS evaluation / smoke-test script.

Runs a forward pass over a small batch from LibriSpeechDataset (mel mode)
and optionally decodes with greedy CTC decoding.

Usage:
    # Smoke test (random weights, no data needed):
    python SqueezeFormer/test.py --smoke_test

    # Evaluate on real data:
    python SqueezeFormer/test.py \
        --data_root /path/to/LibriSpeech \
        --split test-clean \
        --checkpoint /path/to/checkpoint.pt
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2CTCTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset.dataset import LibriSpeechDataset
from SqueezeFormer import Squeezeformer, get_config
from SqueezeFormer.train import collate_fn, build_model


def greedy_ctc_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int):
    """Simple greedy CTC decode: argmax then collapse repeats and remove blank."""
    predictions = log_probs.argmax(dim=-1)  # [B, T]
    decoded = []
    for b in range(predictions.size(0)):
        seq = predictions[b, : lengths[b]].tolist()
        collapsed = []
        prev = None
        for tok in seq:
            if tok != prev:
                collapsed.append(tok)
            prev = tok
        collapsed = [t for t in collapsed if t != blank_id]
        decoded.append(collapsed)
    return decoded


def smoke_test(variant: str = "xs"):
    """Instantiate XS and run a random forward pass to verify shapes."""
    config = get_config(variant)
    model = Squeezeformer(
        num_classes=32,
        input_dim=config.input_dim,
        encoder_dim=config.encoder_dim,
        num_encoder_layers=config.num_encoder_layers,
        reduce_layer_index=config.reduce_layer_index,
        recover_layer_index=config.recover_layer_index,
        num_attention_heads=config.num_attention_heads,
        feed_forward_expansion_factor=config.feed_forward_expansion_factor,
        conv_expansion_factor=config.conv_expansion_factor,
        conv_kernel_size=config.conv_kernel_size,
    )
    model.eval()

    B, T, C = 2, 400, 80
    x = torch.randn(B, T, C)
    lengths = torch.tensor([T, T // 2], dtype=torch.long)

    with torch.no_grad():
        log_probs, out_lengths = model(x, lengths)

    print(f"[smoke_test] SqueezeFormer-{variant.upper()}")
    print(f"  params       : {model.count_parameters():,}")
    print(f"  input shape  : {list(x.shape)}")
    print(f"  output shape : {list(log_probs.shape)}")
    print(f"  out lengths  : {out_lengths.tolist()}")
    print("  PASSED")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
    num_classes = tokenizer.vocab_size

    config = get_config(args.variant)
    model = build_model(config, num_classes).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    model.eval()

    dataset = LibriSpeechDataset(
        path_to_data_root=args.data_root,
        include_splits=[args.split],
        tokenizer=tokenizer,
        train_split=False,
        apply_spec_augment=False,
        mode="mel",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2)

    ctc_loss = nn.CTCLoss(blank=tokenizer.pad_token_id, zero_infinity=True)
    total_loss = 0.0

    with torch.no_grad():
        for mel, lengths, labels, label_lengths in loader:
            mel, lengths, labels, label_lengths = (
                mel.to(device), lengths.to(device), labels.to(device), label_lengths.to(device)
            )
            log_probs, out_lengths = model(mel, lengths)
            loss = ctc_loss(log_probs.permute(1, 0, 2), labels, out_lengths, label_lengths)
            total_loss += loss.item()

    print(f"Eval loss: {total_loss / len(loader):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--variant", type=str, default="xs")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="test-clean")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(args.variant)
    elif args.data_root:
        evaluate(args)
    else:
        print("Provide --smoke_test or --data_root. Run with --help for options.")


if __name__ == "__main__":
    main()
