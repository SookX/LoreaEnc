"""
Precompute mel spectrograms for all LibriSpeech FLAC files.
Saves <uid>.mel.pt (float16) alongside each FLAC file.
Run once; subsequent training runs load from cache instantly.

Usage:
    python dataset/precompute_mels.py --data-root dataset/datasets/librispeech/LibriSpeech --workers 16
"""

import os
import argparse
import glob
import torch
import soundfile as sf
import torchaudio.transforms as T
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Must match train.py constants exactly
MEL_N_FFT  = 400
MEL_HOP    = 160
MEL_N_MELS = 80
SAMPLE_RATE = 16_000

# Build transforms once per process (module-level so workers inherit via fork)
_mel_transform = None

def _get_transform():
    global _mel_transform
    if _mel_transform is None:
        _mel_transform = torch.nn.Sequential(
            T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=MEL_N_MELS,
                             n_fft=MEL_N_FFT, hop_length=MEL_HOP),
            T.AmplitudeToDB(top_db=80.0),
        )
    return _mel_transform


def process_file(flac_path: str) -> tuple[str, bool, str]:
    """Compute and save mel for a single FLAC file. Returns (path, success, msg)."""
    mel_path = flac_path.replace(".flac", ".mel.pt")
    if os.path.exists(mel_path):
        return flac_path, True, "cached"

    try:
        audio_np, sr = sf.read(flac_path, dtype="float32", always_2d=False)
        audio = torch.from_numpy(audio_np).unsqueeze(0)  # [1, T]

        if sr != SAMPLE_RATE:
            import torchaudio
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

        transform = _get_transform()
        with torch.no_grad():
            mel = transform(audio)          # [1, n_mels, T_mel]
            mel = mel.squeeze(0)            # [n_mels, T_mel]
            mel = mel.T                     # [T_mel, n_mels]  — matches model input shape
            # Per-sample normalisation (deterministic, safe to bake in)
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        torch.save(mel.half(), mel_path)    # float16 → ~158 KB per file
        return flac_path, True, "ok"

    except Exception as e:
        return flac_path, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True,
                        help="Root of LibriSpeech (contains train-clean-100 etc.)")
    parser.add_argument("--splits", nargs="+",
                        default=["train-clean-100", "train-clean-360",
                                 "train-other-500", "dev-clean", "test-clean"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    flac_files = []
    for split in args.splits:
        pattern = os.path.join(args.data_root, split, "**", "*.flac")
        found = glob.glob(pattern, recursive=True)
        print(f"  {split}: {len(found):,} files")
        flac_files.extend(found)

    print(f"\nTotal: {len(flac_files):,} files | Workers: {args.workers}")

    already_done = sum(1 for f in flac_files if os.path.exists(f.replace(".flac", ".mel.pt")))
    print(f"Already cached: {already_done:,} | Remaining: {len(flac_files) - already_done:,}\n")

    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_file, f): f for f in flac_files}
        with tqdm(total=len(flac_files), unit="file", dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                path, ok, msg = future.result()
                if not ok:
                    failed.append((path, msg))
                bar.update(1)
                if bar.n % 5000 == 0 and bar.n > 0:
                    bar.set_postfix_str(f"failed={len(failed)}", refresh=True)

    print(f"\nDone. Failed: {len(failed)}")
    if failed:
        for path, msg in failed[:20]:
            print(f"  {os.path.basename(path)}: {msg}")


if __name__ == "__main__":
    main()
