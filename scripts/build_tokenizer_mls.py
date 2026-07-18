"""Train a per-language SentencePiece BPE tokenizer for Multilingual LibriSpeech.

Matches the English recipe's SentencePieceCTCTokenizer contract exactly:
  * pad at id 0  -> used as the CTC blank (SentencePieceCTCTokenizer.pad_token_id = 0)
  * unk at id 1
  * bos/eos disabled (-1)
  * BPE, vocab_size 128 by default

So the resulting model drops straight into train_ctc / finetune via
``--tokenizer-path <model>`` with no code changes.

Usage:
    python scripts/build_tokenizer_mls.py \
        --lang-root dataset/mls/mls_polish \
        --output dataset/mls/mls_polish/bpe128_polish.model

Character coverage defaults to 1.0, which is appropriate for the Latin-script
MLS languages (Polish/Portuguese/etc.); lower it only for large-alphabet langs.
"""

import argparse
import os
import sys
import tempfile

# iter_mls_items lives in the package; make sure the repo root is importable
# whether this is run as a module or a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CausalSpecUnit.data import iter_mls_items  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang-root", required=True,
                    help="Single-language MLS root, e.g. dataset/mls/mls_polish")
    ap.add_argument("--output", required=True,
                    help="Output .model path (a matching .vocab is written alongside)")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--vocab-size", type=int, default=128)
    ap.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    ap.add_argument("--character-coverage", type=float, default=1.0)
    ap.add_argument("--max-sentences", type=int, default=None,
                    help="Cap training sentences (for a quick smoke); default uses all.")
    args = ap.parse_args()

    import sentencepiece as spm

    if not args.output.endswith(".model"):
        raise ValueError("--output must end in .model (SentencePiece appends .model/.vocab)")
    model_prefix = args.output[: -len(".model")]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    # Dump transcripts (one per line) to a temp corpus file for the trainer.
    n = 0
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        corpus_path = tf.name
        for item in iter_mls_items(args.lang_root, [args.train_split]):
            tf.write(item["transcript"].strip() + "\n")
            n += 1
            if args.max_sentences is not None and n >= args.max_sentences:
                break
    print(f"Collected {n} transcript lines from {args.lang_root}/{args.train_split}")
    if n == 0:
        raise SystemExit("No transcripts found — check --lang-root / --train-split.")

    try:
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
            # CTC contract: pad(=blank) at 0, unk at 1, no bos/eos.
            pad_id=0,
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
            pad_piece="<pad>",
            unk_piece="<unk>",
            input_sentence_size=1_000_000,
            shuffle_input_sentence=True,
        )
    finally:
        os.unlink(corpus_path)

    # Sanity: reload via the exact wrapper training uses and round-trip a sample.
    from SqueezeFormer.train import SentencePieceCTCTokenizer
    tok = SentencePieceCTCTokenizer(args.output)
    assert tok.pad_token_id == 0, "pad/blank must be id 0 for the CTC recipe"
    sample = next(iter_mls_items(args.lang_root, [args.train_split]))["transcript"]
    ids = tok.encode(sample)
    print(f"\nTokenizer written: {args.output}  (vocab_size={tok.vocab_size}, pad/blank=0)")
    print(f"  sample text : {sample[:80]!r}")
    print(f"  encoded ids : {ids[:30]}{' ...' if len(ids) > 30 else ''}")
    print(f"  round-trip  : {tok.decode(ids)[:80]!r}")


if __name__ == "__main__":
    main()
