import os
import requests
import tarfile
from tqdm import tqdm



target_folder = "datasets/librispeech"
os.makedirs(target_folder, exist_ok=True)

base_url = "http://www.openslr.org/resources/12"

files = {
    "train-clean-100.tar.gz": f"{base_url}/train-clean-100.tar.gz",
    "train-clean-360.tar.gz": f"{base_url}/train-clean-360.tar.gz",
    "train-other-500.tar.gz": f"{base_url}/train-other-500.tar.gz",
    "dev-clean.tar.gz":        f"{base_url}/dev-clean.tar.gz",
    "test-clean.tar.gz":       f"{base_url}/test-clean.tar.gz",
}


def download_url(url, destination, chunk_size=32768):
    # Get stream
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total_size = r.headers.get("Content-Length")
    if total_size is None:
        total = None
    else:
        total = int(total_size)

    with open(destination, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=os.path.basename(destination)
    ) as bar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

print("🔥 Starting dataset download...")

for fname, url in files.items():
    output_path = os.path.join(target_folder, fname)
    if os.path.exists(output_path):
        print(f"✅ Already downloaded: {fname}")
        continue

    print(f"⬇️ Downloading {fname} from {url}")
    download_url(url, output_path)

print("\n✅ All files downloaded.")

print("\n📦 Extracting .tar.gz files...")

for fname in os.listdir(target_folder):
    if fname.endswith(".tar.gz"):
        full_path = os.path.join(target_folder, fname)
        print(f"➡️ Extracting: {fname}")
        with tarfile.open(full_path, "r:gz") as tar:
            tar.extractall(path=target_folder)

print("\n🎉 DONE! LibriSpeech dataset is ready.")
print(f"📁 Saved in: {os.path.abspath(target_folder)}")
