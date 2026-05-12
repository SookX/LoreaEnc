"""Unzip with a live progress bar. Usage: python unzip_progress.py <zip> <dest>"""
import sys
import zipfile
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Usage: python unzip_progress.py <zipfile> <destination>")
        sys.exit(1)

    zip_path = sys.argv[1]
    dest = Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
        from tqdm import tqdm

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        total_bytes = sum(m.file_size for m in members)

        with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
                  desc="Extracting", ncols=80) as bar:
            for member in members:
                zf.extract(member, dest)
                bar.update(member.file_size)

    print(f"Done — extracted {len(members):,} files to {dest}")


if __name__ == "__main__":
    main()
