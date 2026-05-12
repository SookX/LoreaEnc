import os
import sys
import time
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


def dataset_trace(message):
    if os.environ.get("CSU_TRACE_DATASET", "0") != "1":
        return
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    rank = os.environ.get("RANK", "?")
    print(f"[data-trace {now}] rank={rank} {message}", file=sys.stderr, flush=True)


def iter_librispeech_items(data_root, splits):
    for split in splits:
        split_dir = os.path.join(data_root, split)
        for speaker in sorted(os.listdir(split_dir)):
            speaker_dir = os.path.join(split_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            for chapter in sorted(os.listdir(speaker_dir)):
                chapter_dir = os.path.join(speaker_dir, chapter)
                if not os.path.isdir(chapter_dir):
                    continue
                txt_files = [p for p in os.listdir(chapter_dir) if p.endswith(".txt")]
                if not txt_files:
                    continue
                with open(os.path.join(chapter_dir, txt_files[0]), encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        uid = parts[0]
                        transcript = " ".join(parts[1:])
                        yield {
                            "uid": uid,
                            "audio_path": os.path.join(chapter_dir, uid + ".flac"),
                            "transcript": transcript,
                            "split": split,
                        }


class LogMelExtractor(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80):
        super().__init__()
        self.sample_rate = sample_rate
        self.mels = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=int(sample_rate * 0.025),
            hop_length=int(sample_rate * 0.010),
            f_min=0.0,
            f_max=sample_rate / 2,
            n_mels=n_mels,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        self.amp2db = T.AmplitudeToDB(top_db=80.0)

    def load_audio(self, path):
        audio, sr = torchaudio.load(path, normalize=True)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sample_rate)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio / (audio.abs().amax(dim=-1, keepdim=True) + 1e-9)
        audio = torch.cat([audio[:, :1], audio[:, 1:] - 0.97 * audio[:, :-1]], dim=-1)
        return audio

    @torch.no_grad()
    def forward(self, path):
        audio = self.load_audio(path)
        mel = self.amp2db(self.mels(audio))[0].T
        return mel.float()


def load_cmvn(path):
    state = torch.load(path, map_location="cpu")
    return state["mean"].float(), state["std"].float().clamp_min(1e-5)


def apply_cmvn(mel, mean, std):
    return (mel - mean.to(mel.device)) / std.to(mel.device)


def load_targets(targets_path):
    targets_dir = os.path.dirname(targets_path)
    index_path = os.path.join(targets_dir, "target_index.json")
    if not os.path.isfile(index_path):
        start = time.time()
        dataset_trace(f"monolithic targets load start path={targets_path}")
        targets = torch.load(targets_path, map_location="cpu")
        dataset_trace(f"monolithic targets load done entries={len(targets)} seconds={time.time() - start:.1f}")
        return targets

    start = time.time()
    dataset_trace(f"target index load start path={index_path}")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    uid_to_shard = index["uid_to_shard"]
    shards_dir = os.path.join(targets_dir, index.get("shards_dir", "targets_shards"))
    targets = {}
    for shard_name in sorted(set(uid_to_shard.values())):
        shard_path = os.path.join(shards_dir, shard_name)
        dataset_trace(f"target shard load start path={shard_path}")
        shard = torch.load(shard_path, map_location="cpu")
        targets.update(shard)
    dataset_trace(f"sharded targets load done entries={len(targets)} seconds={time.time() - start:.1f}")
    return targets


class SpecUnitDataset(Dataset):
    def __init__(self, data_root, splits, targets_path, cmvn_path, max_items=None):
        start = time.time()
        dataset_trace(f"scan start data_root={data_root} splits={splits}")
        self.items = list(iter_librispeech_items(data_root, splits))
        dataset_trace(f"scan done items={len(self.items)} seconds={time.time() - start:.1f}")
        if max_items is not None:
            self.items = self.items[:max_items]
        start = time.time()
        dataset_trace(f"targets load start path={targets_path}")
        self.targets = load_targets(targets_path)
        dataset_trace(f"targets load done entries={len(self.targets)} seconds={time.time() - start:.1f}")
        start = time.time()
        dataset_trace(f"cmvn load start path={cmvn_path}")
        self.mean, self.std = load_cmvn(cmvn_path)
        dataset_trace(f"cmvn load done seconds={time.time() - start:.1f}")
        self.extractor = LogMelExtractor()
        start = time.time()
        dataset_trace("filter start")
        self.items = [item for item in self.items if item["uid"] in self.targets]
        dataset_trace(f"filter done items={len(self.items)} seconds={time.time() - start:.1f}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        mel = apply_cmvn(self.extractor(item["audio_path"]), self.mean, self.std)
        target = self.targets[item["uid"]]
        return {
            "uid": item["uid"],
            "mel": mel,
            "z100": target["z100"].long(),
            "z500": target["z500"].long(),
        }


class CTCSpecDataset(Dataset):
    def __init__(self, data_root, splits, tokenizer, cmvn_path=None, train_split=True, max_items=None):
        self.items = list(iter_librispeech_items(data_root, splits))
        if max_items is not None:
            self.items = self.items[:max_items]
        self.tokenizer = tokenizer
        self.extractor = LogMelExtractor()
        self.train_split = train_split
        self.mean = self.std = None
        if cmvn_path is not None:
            self.mean, self.std = load_cmvn(cmvn_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        mel = self.extractor(item["audio_path"])
        if self.mean is not None:
            mel = apply_cmvn(mel, self.mean, self.std)
        labels = torch.tensor(self.tokenizer.encode(item["transcript"]), dtype=torch.long)
        return {
            "uid": item["uid"],
            "mel": mel,
            "labels": labels,
            "transcript": item["transcript"],
        }


def collate_ssl(batch):
    mel = nn.utils.rnn.pad_sequence([b["mel"] for b in batch], batch_first=True)
    lengths = torch.tensor([b["mel"].size(0) for b in batch], dtype=torch.long)
    z100 = nn.utils.rnn.pad_sequence([b["z100"] for b in batch], batch_first=True, padding_value=-100)
    z500 = nn.utils.rnn.pad_sequence([b["z500"] for b in batch], batch_first=True, padding_value=-100)
    target_lengths = torch.tensor([b["z100"].numel() for b in batch], dtype=torch.long)
    return mel, lengths, z100, z500, target_lengths


def collate_ctc(batch):
    mel = nn.utils.rnn.pad_sequence([b["mel"] for b in batch], batch_first=True)
    lengths = torch.tensor([b["mel"].size(0) for b in batch], dtype=torch.long)
    labels = torch.cat([b["labels"] for b in batch])
    label_lengths = torch.tensor([b["labels"].numel() for b in batch], dtype=torch.long)
    return mel, lengths, labels, label_lengths


def collate_eval(batch):
    mel, lengths, labels, label_lengths = collate_ctc(batch)
    transcripts = [b["transcript"] for b in batch]
    return mel, lengths, labels, label_lengths, transcripts
