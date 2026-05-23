import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm

from CausalSpecUnit.data import load_targets


SILENCE_LABELS = {"", "sil", "sp", "spn", "nsn", "<eps>", "<sil>", "silence"}

REQUIRED_METADATA_KEYS = ("chunk_size", "chunk_stride")


def log(message):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[phone-purity {now}] {message}", file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets-dir", type=str, required=True,
                   help="Directory containing metadata.json plus targets.pt or sharded targets.")
    p.add_argument("--textgrid-dir", type=str, required=True,
                   help="Directory containing MFA TextGrid files, searched recursively by UID stem.")
    p.add_argument("--tier", type=str, default="phones",
                   help="Preferred TextGrid interval tier name.")
    p.add_argument("--frame-hop", type=float, default=0.010,
                   help="Spectrogram frame hop in seconds (must match hop_length / sample_rate used "
                        "in generate_targets.py; default 160/16000 = 0.010 s).")
    p.add_argument("--exclude-silence", action="store_true",
                   help="Exclude chunks whose dominant phone is a silence/noise label.")
    p.add_argument("--max-utterances", type=int, default=None)
    p.add_argument("--output", type=str, default=None,
                   help="Optional .npz path to save cluster/phone pairs and metric summaries.")
    return p.parse_args()


def read_metadata(targets_dir):
    path = os.path.join(targets_dir, "metadata.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing metadata.json in {targets_dir}")
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    missing = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
    if missing:
        raise KeyError(f"metadata.json is missing required keys: {missing}")
    return meta


def normalize_phone(label):
    label = label.strip().lower()
    label = re.sub(r"\d+$", "", label)  # strip ARPABET stress digits: AH0 -> AH
    return label


def build_textgrid_index(textgrid_dir, progress=True):
    index = {}
    n_dirs = n_files = 0
    iterator = os.walk(textgrid_dir)
    if progress:
        iterator = tqdm(iterator, desc="Index TextGrids", unit="dir")
    for root, _, files in iterator:
        n_dirs += 1
        for name in files:
            if name.lower().endswith(".textgrid"):
                uid = os.path.splitext(name)[0]
                index[uid] = os.path.join(root, name)
                n_files += 1
        if progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(files=n_files)
    log(f"TextGrid index done: dirs={n_dirs:,} files={n_files:,}")
    return index


def parse_textgrid(path, preferred_tier="phones"):
    """
    Parse MFA long-format TextGrids.

    Returns a list of (xmin, xmax, phone) tuples for the best matching tier.
    Raises ValueError with a descriptive message if the file cannot be parsed.
    """
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f]

    tiers = []
    current = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("class =") and "IntervalTier" in line:
            if current is not None:
                tiers.append(current)
            current = {"name": "", "intervals": []}
        elif current is not None and line.startswith("name ="):
            current["name"] = line.split("=", 1)[1].strip().strip('"').lower()
        elif current is not None and line.startswith("intervals ["):
            xmin = xmax = text = None
            j = i + 1
            while j < len(lines):
                l = lines[j]
                if l.startswith("intervals [") or l.startswith("item ["):
                    break
                if l.startswith("xmin ="):
                    xmin = float(l.split("=", 1)[1].strip())
                elif l.startswith("xmax ="):
                    xmax = float(l.split("=", 1)[1].strip())
                elif l.startswith("text ="):
                    text = l.split("=", 1)[1].strip().strip('"')
                j += 1
            if xmin is not None and xmax is not None and text is not None:
                current["intervals"].append((xmin, xmax, normalize_phone(text)))
            else:
                warnings.warn(f"Incomplete interval at line {i} in {path} — skipping.")
            i = j - 1
        i += 1
    if current is not None:
        tiers.append(current)

    if not tiers:
        raise ValueError(f"No IntervalTier entries found in {path}. "
                         "Check that MFA produced long-format TextGrids.")

    preferred = preferred_tier.lower()
    phone_tier_names = {"phone", "phones", "phoneme", "phonemes"}
    for tier in tiers:
        if tier["name"] == preferred:
            if not tier["intervals"]:
                warnings.warn(f"Preferred tier '{preferred}' has no intervals in {path}.")
            return tier["intervals"]
    for tier in tiers:
        if tier["name"] in phone_tier_names:
            return tier["intervals"]

    # Fall back to last tier but warn
    fallback = tiers[-1]
    warnings.warn(
        f"No tier named '{preferred}' or known phone-tier name in {path}. "
        f"Falling back to last tier: '{fallback['name']}'."
    )
    return fallback["intervals"]


def dominant_phone(intervals, start_time, end_time):
    """Return the phone label with the most overlap in [start_time, end_time)."""
    overlaps = Counter()
    for phone_start, phone_end, phone in intervals:
        overlap = max(0.0, min(end_time, phone_end) - max(start_time, phone_start))
        if overlap > 0:
            overlaps[phone] += overlap
    if not overlaps:
        return None
    return overlaps.most_common(1)[0][0]


def dominant_phone_linear(intervals, start_time, end_time, start_pos):
    """Return dominant phone plus next interval search position.

    Chunks are visited in increasing time order, so reusing the previous
    interval index avoids scanning every TextGrid interval for every chunk.
    """
    n = len(intervals)
    while start_pos < n and intervals[start_pos][1] <= start_time:
        start_pos += 1

    overlaps = Counter()
    j = start_pos
    while j < n:
        phone_start, phone_end, phone = intervals[j]
        if phone_start >= end_time:
            break
        overlap = max(0.0, min(end_time, phone_end) - max(start_time, phone_start))
        if overlap > 0:
            overlaps[phone] += overlap
        j += 1

    if not overlaps:
        return None, start_pos
    return overlaps.most_common(1)[0][0], start_pos


def cluster_purity(cluster_ids, phone_labels):
    by_cluster = defaultdict(Counter)
    for cid, phone in zip(cluster_ids, phone_labels):
        by_cluster[int(cid)][phone] += 1

    total = correct = 0
    per_cluster = {}
    for cid, counts in by_cluster.items():
        n = sum(counts.values())
        majority_phone, majority_count = counts.most_common(1)[0]
        total += n
        correct += majority_count
        per_cluster[cid] = {
            "majority_phone": majority_phone,
            "purity": majority_count / n,
            "count": n,
        }
    return correct / max(total, 1), per_cluster


def collect_pairs(targets, textgrid_index, metadata, tier, frame_hop, exclude_silence, max_utterances, progress=True):
    chunk_size = int(metadata["chunk_size"])
    chunk_stride = int(metadata["chunk_stride"])
    log(
        "Collecting cluster/phone pairs "
        f"targets={len(targets):,} textgrids={len(textgrid_index):,} "
        f"chunk_size={chunk_size} chunk_stride={chunk_stride} "
        f"exclude_silence={exclude_silence}"
    )

    z100_all, z500_all, phones_all = [], [], []
    uids_used, missing_textgrid, empty_intervals = [], [], []

    items = targets.items()
    total = len(targets)
    if max_utterances is not None:
        total = min(total, max_utterances)
    iterator = tqdm(items, total=total, desc="Match chunks", unit="utt", disable=not progress)

    for uid, target in iterator:
        if max_utterances is not None and len(uids_used) >= max_utterances:
            break

        if uid not in textgrid_index:
            missing_textgrid.append(uid)
            continue

        try:
            intervals = parse_textgrid(textgrid_index[uid], preferred_tier=tier)
        except ValueError as e:
            warnings.warn(str(e))
            empty_intervals.append(uid)
            continue

        if not intervals:
            empty_intervals.append(uid)
            continue

        z100 = target["z100"].numpy() if isinstance(target["z100"], torch.Tensor) else np.asarray(target["z100"])
        z500 = target["z500"].numpy() if isinstance(target["z500"], torch.Tensor) else np.asarray(target["z500"])
        n_chunks = min(len(z100), len(z500))

        n_before = len(phones_all)
        interval_pos = 0
        for idx in range(n_chunks):
            start_time = idx * chunk_stride * frame_hop
            end_time = (idx * chunk_stride + chunk_size) * frame_hop
            phone, interval_pos = dominant_phone_linear(intervals, start_time, end_time, interval_pos)
            if phone is None:
                continue
            if exclude_silence and phone in SILENCE_LABELS:
                continue
            z100_all.append(int(z100[idx]))
            z500_all.append(int(z500[idx]))
            phones_all.append(phone)

        if len(phones_all) > n_before:
            uids_used.append(uid)
        else:
            empty_intervals.append(uid)

        if progress and len(uids_used) % 1000 == 0 and len(uids_used) > 0:
            iterator.set_postfix(
                used=len(uids_used),
                pairs=len(phones_all),
                missing=len(missing_textgrid),
                empty=len(empty_intervals),
            )

    return (
        np.asarray(z100_all, dtype=np.int64),
        np.asarray(z500_all, dtype=np.int64),
        np.asarray(phones_all, dtype=object),
        uids_used,
        missing_textgrid,
        empty_intervals,
    )


def summarize(name, cluster_ids, phones):
    purity, per_cluster = cluster_purity(cluster_ids, phones)
    nmi = normalized_mutual_info_score(phones, cluster_ids)
    num_clusters = len(set(cluster_ids.tolist()))
    num_phones = len(set(phones.tolist()))
    print(f"\n{name}:")
    print(f"  chunks         : {len(cluster_ids):,}")
    print(f"  active clusters: {num_clusters}")
    print(f"  phones         : {num_phones}")
    print(f"  purity         : {purity:.4f}")
    print(f"  NMI            : {nmi:.4f}")
    return {"purity": purity, "nmi": nmi, "active_clusters": num_clusters,
            "num_phones": num_phones, "per_cluster": per_cluster}


def main():
    args = parse_args()
    log(f"Starting with args={vars(args)}")

    if args.frame_hop > 1.0:
        raise ValueError(
            f"--frame-hop={args.frame_hop} looks like samples, not seconds. "
            "Pass the value in seconds (e.g. 0.010 for a 160-sample hop at 16 kHz)."
        )

    log(f"Reading metadata from {args.targets_dir}")
    metadata = read_metadata(args.targets_dir)
    log(
        "Metadata loaded: "
        + json.dumps({k: metadata.get(k) for k in ("chunk_size", "chunk_stride", "k_coarse", "k_fine", "num_target_utterances")})
    )

    targets_path = os.path.join(args.targets_dir, "targets.pt")
    index_path = os.path.join(args.targets_dir, "target_index.json")
    if os.path.isfile(index_path):
        log(f"Loading sharded targets via {index_path}")
    else:
        log(f"Loading monolithic targets from {targets_path}")
    t0 = time.time()
    targets = load_targets(os.path.join(args.targets_dir, "targets.pt"))
    log(f"Targets loaded: entries={len(targets):,} seconds={time.time() - t0:.1f}")

    log(f"Indexing TextGrids under {args.textgrid_dir}")
    t0 = time.time()
    textgrid_index = build_textgrid_index(args.textgrid_dir)
    print(f"Indexed {len(textgrid_index):,} TextGrid files", flush=True)
    log(f"TextGrid indexing elapsed seconds={time.time() - t0:.1f}")

    t0 = time.time()
    z100, z500, phones, used_uids, missing_tg, empty_intervals = collect_pairs(
        targets=targets,
        textgrid_index=textgrid_index,
        metadata=metadata,
        tier=args.tier,
        frame_hop=args.frame_hop,
        exclude_silence=args.exclude_silence,
        max_utterances=args.max_utterances,
    )
    log(f"Pair collection elapsed seconds={time.time() - t0:.1f}")

    print(f"Utterances with valid chunks : {len(used_uids):,}", flush=True)
    print(f"Missing TextGrid             : {len(missing_tg):,}", flush=True)
    print(f"Empty/unparseable TextGrid   : {len(empty_intervals):,}", flush=True)
    print(f"Silence excluded             : {args.exclude_silence}", flush=True)

    if len(phones) == 0:
        raise RuntimeError(
            "No cluster/phone pairs collected. "
            "Check --textgrid-dir, UID naming, and that MFA produced long-format TextGrids."
        )

    phone_counts = Counter(phones.tolist())
    print(f"\nTop phones: {phone_counts.most_common(15)}", flush=True)

    summary100 = summarize("K=100", z100, phones)
    summary500 = summarize("K=500", z500, phones)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(
            args.output,
            z100=z100,
            z500=z500,
            phones=phones,
            used_uids=np.asarray(used_uids, dtype=object),
            missing_uids=np.asarray(missing_tg, dtype=object),
            k100_purity=summary100["purity"],
            k100_nmi=summary100["nmi"],
            k500_purity=summary500["purity"],
            k500_nmi=summary500["nmi"],
        )
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
