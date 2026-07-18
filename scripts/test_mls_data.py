"""Sanity-check MLS enumeration + tokenizer wiring on the cluster (data must be
on disk). Not a unit test in CI — a quick real-data smoke:

    python scripts/test_mls_data.py --lang-root dataset/mls/mls_polish
    python scripts/test_mls_data.py --lang-root dataset/mls/mls_polish \
        --tokenizer dataset/mls/mls_polish/bpe128_polish.model

Checks, per split:
  * transcripts parse and yield items
  * the first --check-audio audio paths actually exist and are 16 kHz mono
  * a sample-based train-hours estimate (so we know the pretraining pool size)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CausalSpecUnit.data import iter_mls_items, audio_duration_seconds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang-root", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--check-audio", type=int, default=20,
                    help="How many audio files per split to actually open.")
    ap.add_argument("--hours-sample", type=int, default=300,
                    help="Utterances to sample for the train-hours estimate.")
    ap.add_argument("--tokenizer", default=None,
                    help="Optional trained .model to round-trip a few transcripts through.")
    args = ap.parse_args()

    ok = True
    for split in args.splits:
        items = list(iter_mls_items(args.lang_root, [split]))
        n = len(items)
        print(f"\n[{split}] {n} utterances")
        if n == 0:
            print("  !! no items"); ok = False; continue

        # audio existence + format on the first K
        bad = 0; srs = set(); chans = set()
        for it in items[: args.check_audio]:
            if not os.path.isfile(it["audio_path"]):
                bad += 1
                if bad <= 3:
                    print(f"  MISSING: {it['audio_path']}")
                continue
            import torchaudio
            info = torchaudio.info(it["audio_path"])
            srs.add(info.sample_rate); chans.add(info.num_channels)
        if bad:
            print(f"  !! {bad}/{args.check_audio} audio paths missing"); ok = False
        else:
            print(f"  audio OK: {args.check_audio} files exist | sample_rate(s)={srs} channels={chans}")
            if srs != {16000}:
                print(f"  !! expected 16 kHz, got {srs}"); ok = False

        # hours estimate (sample) — only meaningful for train
        sample = items[: args.hours_sample]
        secs = []
        for it in sample:
            try:
                secs.append(audio_duration_seconds(it["audio_path"]))
            except Exception as exc:
                print(f"  dur read failed {it['uid']}: {exc}")
        if secs:
            avg = sum(secs) / len(secs)
            est_hours = avg * n / 3600.0
            print(f"  avg dur={avg:.2f}s over {len(secs)} sampled -> est {split} hours ~= {est_hours:.0f}h")

    if args.tokenizer:
        from SqueezeFormer.train import SentencePieceCTCTokenizer
        tok = SentencePieceCTCTokenizer(args.tokenizer)
        print(f"\n[tokenizer] {args.tokenizer}  vocab={tok.vocab_size} pad/blank={tok.pad_token_id}")
        for it in list(iter_mls_items(args.lang_root, ["train"]))[:3]:
            ids = tok.encode(it["transcript"])
            rt = tok.decode(ids)
            print(f"  {it['transcript'][:50]!r} -> {len(ids)} ids -> {rt[:50]!r}")
        if tok.pad_token_id != 0:
            print("  !! pad/blank is not 0 — CTC recipe expects 0"); ok = False

    print("\n" + ("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
