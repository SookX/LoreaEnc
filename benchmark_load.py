import sys
print("Starting...", flush=True)

import glob
print("glob imported", flush=True)

import torch
print("torch imported", flush=True)

from concurrent.futures import ThreadPoolExecutor, as_completed
print("concurrent.futures imported", flush=True)

from tqdm import tqdm
print("tqdm imported", flush=True)

DATA_ROOT = "dataset/datasets/librispeech/LibriSpeech"
SPLITS    = ["train-clean-100", "train-clean-360", "train-other-500", "dev-clean"]
WORKERS   = 16

print("Scanning files...", flush=True)
files = []
for split in SPLITS:
    found = glob.glob(f"{DATA_ROOT}/{split}/**/*.mel.pt", recursive=True)
    print(f"  {split}: {len(found):,} files", flush=True)
    files += found
print(f"Total: {len(files):,} mel files", flush=True)

total_bytes = 0

def load_one(path):
    t = torch.load(path, weights_only=True)
    return t.nbytes

print(f"\nLoading with {WORKERS} threads...", flush=True)
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(load_one, f): f for f in files}
    with tqdm(total=len(files), unit="file", dynamic_ncols=True) as bar:
        for future in as_completed(futures):
            total_bytes += future.result()
            bar.update(1)
            bar.set_postfix_str(f"{total_bytes/1e9:.1f} GB", refresh=False)

print(f"\nDone: {total_bytes/1e9:.1f} GB total", flush=True)
