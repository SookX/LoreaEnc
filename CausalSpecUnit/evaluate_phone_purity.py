import argparse
import json
import os
import re
import warnings
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score


SILENCE_LABELS = {"", "sil", "sp", "spn", "nsn", "<eps>", "<sil>", "silence"}

REQUIRED_METADATA_KEYS = ("chunk_size", "chunk_stride")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets-dir", type=str, required=True,
                   help="Directory containing targets.pt and metadata.json from generate_targets.py.")
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


def build_textgrid_index(textgrid_dir):
    index = {}
    for root, _, files in os.walk(textgrid_dir):
        for name in files:
            if name.lower().endswith(".textgrid"):
                uid = os.path.splitext(name)[0]
                index[uid] = os.path.join(root, name)
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


def collect_pairs(targets, textgrid_index, metadata, tier, frame_hop, exclude_silence, max_utterances):
    chunk_size = int(metadata["chunk_size"])
    chunk_stride = int(metadata["chunk_stride"])

    z100_all, z500_all, phones_all = [], [], []
    uids_used, missing_textgrid, empty_intervals = [], [], []

    for uid, target in targets.items():
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
        for idx in range(n_chunks):
            start_time = idx * chunk_stride * frame_hop
            end_time = (idx * chunk_stride + chunk_size) * frame_hop
            phone = dominant_phone(intervals, start_time, end_time)
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

    if args.frame_hop > 1.0:
        raise ValueError(
            f"--frame-hop={args.frame_hop} looks like samples, not seconds. "
            "Pass the value in seconds (e.g. 0.010 for a 160-sample hop at 16 kHz)."
        )

    metadata = read_metadata(args.targets_dir)
    targets = torch.load(
        os.path.join(args.targets_dir, "targets.pt"),
        map_location="cpu",
        weights_only=False,
    )
    textgrid_index = build_textgrid_index(args.textgrid_dir)
    print(f"Indexed {len(textgrid_index):,} TextGrid files")

    z100, z500, phones, used_uids, missing_tg, empty_intervals = collect_pairs(
        targets=targets,
        textgrid_index=textgrid_index,
        metadata=metadata,
        tier=args.tier,
        frame_hop=args.frame_hop,
        exclude_silence=args.exclude_silence,
        max_utterances=args.max_utterances,
    )

    print(f"Utterances with valid chunks : {len(used_uids):,}")
    print(f"Missing TextGrid             : {len(missing_tg):,}")
    print(f"Empty/unparseable TextGrid   : {len(empty_intervals):,}")
    print(f"Silence excluded             : {args.exclude_silence}")

    if len(phones) == 0:
        raise RuntimeError(
            "No cluster/phone pairs collected. "
            "Check --textgrid-dir, UID naming, and that MFA produced long-format TextGrids."
        )

    phone_counts = Counter(phones.tolist())
    print(f"\nTop phones: {phone_counts.most_common(15)}")

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
