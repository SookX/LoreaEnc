import os
from collections import defaultdict

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


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


class SpecUnitDataset(Dataset):
    def __init__(self, data_root, splits, targets_path, cmvn_path, max_items=None):
        self.items = list(iter_librispeech_items(data_root, splits))
        if max_items is not None:
            self.items = self.items[:max_items]
        self.targets = torch.load(targets_path, map_location="cpu")
        self.mean, self.std = load_cmvn(cmvn_path)
        self.extractor = LogMelExtractor()
        self.items = [item for item in self.items if item["uid"] in self.targets]

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

