#!/usr/bin/env python3
"""Download and prepare the official Libri-Light limited-supervision splits.

The CausalSpecUnit training code reads LibriSpeech-style split directories:

    <data-root>/<split>/<speaker>/<chapter>/<uid>.flac
    <data-root>/<split>/<speaker>/<chapter>/<chapter>.txt

The official Libri-Light limited-supervision archive has an extra split
prefix, e.g. ``librispeech_finetuning/1h/0/clean/...``. This script keeps
the official extracted archive intact, then creates prepared split
directories with symlinks and regenerated chapter transcript files:

    librilight_10min
    librilight_1h
    librilight_10h

It also downloads the standard LibriSpeech dev/test archives used by the
project and verifies checksums where official checksums are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path


LIBRILIGHT_URL = "https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz"
LIBRILIGHT_SHA256 = "5d1efdc777b548194d7e09ba89126e2188026df9fd57aa57eb14408d2b2342af"
LIBRILIGHT_ARCHIVE = "librispeech_finetuning.tgz"
LIBRILIGHT_DIR = "librispeech_finetuning"

LIBRISPEECH_BASE_URL = "https://www.openslr.org/resources/12"
LIBRISPEECH_EVAL_ARCHIVES = [
    "dev-clean.tar.gz",
    "dev-other.tar.gz",
    "test-clean.tar.gz",
    "test-other.tar.gz",
]

SUBSET_FOLDERS = {
    "librilight_10min": ["1h/0"],
    "librilight_1h": ["1h/0", "1h/1", "1h/2", "1h/3", "1h/4", "1h/5"],
    "librilight_10h": ["1h/0", "1h/1", "1h/2", "1h/3", "1h/4", "1h/5", "9h"],
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log(f"exists: {dest}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"download: {url} -> {dest}")
    curl = shutil.which("curl")
    if curl:
        subprocess.run([curl, "-L", "--fail", "--continue-at", "-", "--output", str(tmp), url], check=True)
    else:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
    tmp.rename(dest)


def safe_extract_tar(archive: Path, dest: Path) -> None:
    log(f"test archive integrity: {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"Refusing unsafe tar member: {member.name}")
        log(f"extract: {archive} -> {dest}")
        tar.extractall(dest)


def read_openslr_md5s(raw_root: Path) -> dict[str, str]:
    md5_path = raw_root / "md5sum.txt"
    download(f"{LIBRISPEECH_BASE_URL}/md5sum.txt", md5_path)
    md5s = {}
    for line in md5_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            md5s[parts[-1]] = parts[0]
    return md5s


def ensure_librilight(raw_root: Path) -> Path:
    archive = raw_root / LIBRILIGHT_ARCHIVE
    extracted = raw_root / LIBRILIGHT_DIR
    download(LIBRILIGHT_URL, archive)
    digest = sha256_file(archive)
    if digest != LIBRILIGHT_SHA256:
        raise RuntimeError(
            f"SHA256 mismatch for {archive}: got {digest}, expected {LIBRILIGHT_SHA256}"
        )
    log(f"verified SHA256: {archive}")
    if not extracted.is_dir():
        safe_extract_tar(archive, raw_root)
    else:
        log(f"already extracted: {extracted}")
    return extracted


def ensure_librispeech_eval(data_root: Path, raw_root: Path) -> None:
    md5s = read_openslr_md5s(raw_root)
    for name in LIBRISPEECH_EVAL_ARCHIVES:
        archive = raw_root / name
        split = name.replace(".tar.gz", "")
        download(f"{LIBRISPEECH_BASE_URL}/{name}", archive)
        expected = md5s.get(name)
        if expected:
            digest = md5_file(archive)
            if digest != expected:
                raise RuntimeError(f"MD5 mismatch for {archive}: got {digest}, expected {expected}")
            log(f"verified MD5: {archive}")
        else:
            log(f"no MD5 found for {name}; archive integrity will be checked by tarfile")
        if not (data_root / split).is_dir():
            safe_extract_tar(archive, data_root.parent)
        else:
            log(f"already extracted: {data_root / split}")


def parse_transcript_files(chapter_dir: Path) -> dict[str, str]:
    transcripts = {}
    for txt in chapter_dir.glob("*.trans.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]
    return transcripts


def relative_subset(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return "/".join(rel.parts[:2]) if rel.parts[0] == "1h" else rel.parts[0]


def collect_official_items(extracted: Path, folders: list[str]) -> list[dict[str, str]]:
    items = []
    folder_set = set(folders)
    flacs = list(extracted.glob("1h/*/*/*/*/*.flac")) + list(extracted.glob("9h/*/*/*/*.flac"))
    for flac in sorted(flacs):
        subset_key = relative_subset(flac, extracted)
        if subset_key not in folder_set:
            continue
        uid = flac.stem
        speaker = flac.parent.parent.name
        chapter = flac.parent.name
        transcripts = parse_transcript_files(flac.parent)
        transcript = transcripts.get(uid)
        if transcript is None:
            raise RuntimeError(f"Missing transcript for {uid} in {flac.parent}")
        items.append(
            {
                "uid": uid,
                "speaker": speaker,
                "chapter": chapter,
                "transcript": transcript,
                "official_subset_path": subset_key,
                "source_audio": str(flac),
            }
        )
    if not items:
        raise RuntimeError(f"No FLAC files found for official folders {folders} under {extracted}")
    return items


def link_or_copy(src: Path, dst: Path, copy_audio: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_audio:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.relpath(src, start=dst.parent), dst)


def prepare_split(data_root: Path, manifest_root: Path, split: str, items: list[dict[str, str]], copy_audio: bool) -> dict:
    split_dir = data_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    by_chapter: dict[tuple[str, str], list[dict[str, str]]] = {}
    manifest_path = manifest_root / f"{split}.jsonl"
    for item in items:
        src = Path(item["source_audio"])
        dst = split_dir / item["speaker"] / item["chapter"] / src.name
        link_or_copy(src, dst, copy_audio=copy_audio)
        item = dict(item)
        item["audio_path"] = str(dst)
        by_chapter.setdefault((item["speaker"], item["chapter"]), []).append(item)

    for (speaker, chapter), chapter_items in by_chapter.items():
        txt_path = split_dir / speaker / chapter / f"{chapter}.txt"
        lines = [f"{it['uid']} {it['transcript']}" for it in sorted(chapter_items, key=lambda x: x["uid"])]
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for item in sorted(items, key=lambda x: x["uid"]):
            src = Path(item["source_audio"])
            audio_path = split_dir / item["speaker"] / item["chapter"] / src.name
            f.write(
                json.dumps(
                    {
                        "uid": item["uid"],
                        "speaker": item["speaker"],
                        "chapter": item["chapter"],
                        "transcript": item["transcript"],
                        "audio_path": str(audio_path),
                        "official_subset_path": item["official_subset_path"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    return {
        "split": split,
        "utterances": len(items),
        "speakers": len({it["speaker"] for it in items}),
        "chapters": len({(it["speaker"], it["chapter"]) for it in items}),
        "manifest_path": str(manifest_path),
        "split_dir": str(split_dir),
        "official_folders": SUBSET_FOLDERS[split],
    }


def audio_duration_seconds(path: Path) -> float:
    with path.open("rb") as f:
        if f.read(4) != b"fLaC":
            raise RuntimeError(f"Not a FLAC file: {path}")
        while True:
            header = f.read(4)
            if len(header) != 4:
                raise RuntimeError(f"Missing FLAC STREAMINFO block: {path}")
            is_last = bool(header[0] & 0x80)
            block_type = header[0] & 0x7F
            block_len = int.from_bytes(header[1:4], "big")
            payload = f.read(block_len)
            if block_type == 0:
                if len(payload) < 18:
                    raise RuntimeError(f"Invalid FLAC STREAMINFO block: {path}")
                packed = int.from_bytes(payload[10:18], "big")
                sample_rate = (packed >> 44) & 0xFFFFF
                total_samples = packed & ((1 << 36) - 1)
                if sample_rate <= 0:
                    raise RuntimeError(f"Invalid FLAC sample rate in {path}")
                return float(total_samples) / float(sample_rate)
            if is_last:
                raise RuntimeError(f"Missing FLAC STREAMINFO block: {path}")


def add_duration_stats(stats: dict, manifest_path: Path) -> dict:
    total = 0.0
    count = 0
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total += audio_duration_seconds(Path(row["audio_path"]))
            count += 1
    stats["duration_seconds"] = total
    stats["duration_hours"] = total / 3600.0
    stats["duration_minutes"] = total / 60.0
    stats["duration_utterances_checked"] = count
    return stats


def smoke_test(data_root: Path, split: str) -> None:
    split_dir = data_root / split
    txt_path = next(split_dir.glob("*/*/*.txt"))
    first_line = txt_path.read_text(encoding="utf-8").splitlines()[0]
    uid, transcript = first_line.split(maxsplit=1)
    audio_path = txt_path.parent / f"{uid}.flac"
    duration = audio_duration_seconds(audio_path)
    if not transcript:
        raise RuntimeError(f"Empty transcript in smoke test for {split}")
    log(
        "smoke ok "
        f"split={split} uid={uid} duration={duration:.2f}s "
        f"transcript={transcript[:80]!r}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="dataset/datasets/librispeech/LibriSpeech")
    p.add_argument("--raw-root", default="dataset/datasets/librilight")
    p.add_argument("--manifest-root", default="dataset/manifests/librilight")
    p.add_argument("--stats-path", default="dataset/manifests/librilight/stats.json")
    p.add_argument("--copy-audio", action="store_true", help="Copy FLACs instead of symlinking them.")
    p.add_argument("--skip-download", action="store_true", help="Use already downloaded archives.")
    p.add_argument("--skip-librispeech-eval", action="store_true")
    p.add_argument("--skip-duration-stats", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_root)
    raw_root = Path(args.raw_root)
    manifest_root = Path(args.manifest_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    if args.skip_download:
        extracted = raw_root / LIBRILIGHT_DIR
        if not extracted.is_dir():
            raise RuntimeError(f"--skip-download set but missing {extracted}")
    else:
        extracted = ensure_librilight(raw_root)
        if not args.skip_librispeech_eval:
            ensure_librispeech_eval(data_root, raw_root)

    all_stats = {}
    for split, folders in SUBSET_FOLDERS.items():
        log(f"prepare {split} from official folders: {folders}")
        items = collect_official_items(extracted, folders)
        stats = prepare_split(data_root, manifest_root, split, items, copy_audio=args.copy_audio)
        if not args.skip_duration_stats:
            stats = add_duration_stats(stats, Path(stats["manifest_path"]))
        all_stats[split] = stats
        log(json.dumps(stats, indent=2, sort_keys=True))

    stats_path = Path(args.stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(all_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"wrote stats: {stats_path}")

    if args.smoke_test:
        for split in SUBSET_FOLDERS:
            smoke_test(data_root, split)


if __name__ == "__main__":
    main()
