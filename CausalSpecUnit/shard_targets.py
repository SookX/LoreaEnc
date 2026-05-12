import argparse
import json
import os

import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets-dir", type=str, required=True)
    p.add_argument("--num-shards", type=int, default=64)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    targets_path = os.path.join(args.targets_dir, "targets.pt")
    index_path = os.path.join(args.targets_dir, "target_index.json")
    shards_dir = os.path.join(args.targets_dir, "targets_shards")

    if os.path.isfile(index_path) and os.path.isdir(shards_dir) and not args.force:
        print(f"Sharded targets already exist in {args.targets_dir}")
        return
    if not os.path.isfile(targets_path):
        raise FileNotFoundError(f"Missing monolithic targets file: {targets_path}")

    os.makedirs(shards_dir, exist_ok=True)
    print(f"Loading monolithic targets: {targets_path}")
    targets = torch.load(targets_path, map_location="cpu")
    uids = sorted(targets)
    if not uids:
        raise ValueError(f"No targets found in {targets_path}")

    num_shards = max(1, min(args.num_shards, len(uids)))
    shard_size = (len(uids) + num_shards - 1) // num_shards
    index = {}

    print(f"Writing {num_shards} target shards to {shards_dir}")
    for shard_id in tqdm(range(num_shards), desc="shard targets"):
        start = shard_id * shard_size
        end = min(start + shard_size, len(uids))
        shard_uids = uids[start:end]
        if not shard_uids:
            continue
        shard = {uid: targets[uid] for uid in shard_uids}
        shard_name = f"targets_{shard_id:04d}.pt"
        torch.save(shard, os.path.join(shards_dir, shard_name))
        for uid in shard_uids:
            index[uid] = shard_name

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_targets": len(index),
                "num_shards": num_shards,
                "shards_dir": "targets_shards",
                "uid_to_shard": index,
            },
            f,
        )
    print(f"Wrote target index: {index_path}")


if __name__ == "__main__":
    main()
