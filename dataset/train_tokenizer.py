"""
Train a SentencePiece BPE tokenizer (vocab=128) on LibriSpeech transcripts.
Blank token = pad_id = 0  →  used as CTC blank.

Usage:
    python dataset/train_tokenizer.py \
        --data-root dataset/datasets/librispeech/LibriSpeech \
        --output    dataset/bpe128
"""
import os, argparse, glob
import sentencepiece as spm

def collect_transcripts(data_root, splits, out_txt):
    lines = []
    for split in splits:
        for txt_file in glob.glob(os.path.join(data_root, split, "*", "*", "*.txt")):
            with open(txt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        lines.append(" ".join(parts[1:]))
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    print(f"Collected {len(lines):,} utterances → {out_txt}")
    return len(lines)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output",    default="dataset/bpe128",
                   help="Prefix for .model and .vocab files")
    p.add_argument("--splits", nargs="+",
                   default=["train-clean-100", "train-clean-360", "train-other-500"])
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    txt_path = args.output + "_transcripts.txt"
    collect_transcripts(args.data_root, args.splits, txt_path)

    spm.SentencePieceTrainer.train(
        input=txt_path,
        model_prefix=args.output,
        vocab_size=128,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,        # ← CTC blank
        unk_id=1,
        bos_id=-1,       # no BOS
        eos_id=-1,       # no EOS
        normalization_rule_name="identity",
        input_sentence_size=500000,
        shuffle_input_sentence=True,
    )
    print(f"Saved: {args.output}.model  {args.output}.vocab")

    # Quick sanity check
    sp = spm.SentencePieceProcessor(model_file=args.output + ".model")
    sample = "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES"
    ids = sp.encode(sample)
    dec = sp.decode(ids)
    print(f"Vocab size : {sp.get_piece_size()}")
    print(f"Blank id   : {sp.pad_id()}  (CTC blank)")
    print(f"Sample enc : {ids}")
    print(f"Sample dec : {dec}")

if __name__ == "__main__":
    main()
