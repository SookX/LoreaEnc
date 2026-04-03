"""
dataset/dataset.py
──────────────────
LibriSpeechDataset with two loading modes:
  - Filesystem walker (default): scans directory structure
  - Manifest mode: loads from a JSON file produced by dataset/splits.py
                   Use this for all baseline and Lorea training runs so
                   every model sees the exact same labeled utterances.
"""
import os
import json
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from typing import List, Optional
from transformers import Wav2Vec2CTCTokenizer


class ConformerSpecAugment(torch.nn.Module):
    def __init__(self,
                 time_mask_param=40,
                 freq_mask_param=30,
                 num_time_masks=2,
                 num_freq_masks=2,
                 time_warp=False):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.time_warp = time_warp

        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        if time_warp:
            self.time_warp_tf = torchaudio.transforms.TimeStretch()

    def forward(self, spec):
        out = spec.clone()
        if self.time_warp:
            out = self.time_warp_tf(out)
        for _ in range(self.num_freq_masks):
            out = self.freq_mask(out)
        for _ in range(self.num_time_masks):
            out = self.time_mask(out)
        return out


class LibriSpeechDataset(Dataset):
    """
    LibriSpeech dataset.

    Loading priority:
      1. manifest_path (JSON from dataset/splits.py) — preferred for training.
         Ensures bitwise-identical data splits between baselines and Lorea.
      2. include_splits filesystem walker — used for pre-training (full 960 h)
         where no manifest is needed.

    mode:
      "waveform" — returns raw float32 audio; used by all HF wav2vec2/HuBERT scripts.
      "mel"      — returns 80-dim log-mel spectrogram; used by Lorea's own encoder.
    """

    def __init__(
        self,
        path_to_data_root: str,
        include_splits: List[str] = None,
        manifest_path: Optional[str] = None,
        sampling_rate: int = 16_000,
        num_audio_channels: int = 1,
        tokenizer=None,
        train_split: bool = True,
        apply_spec_augment: bool = True,
        apply_audio_augment: bool = True,
        mode: str = "mel",
    ):
        if manifest_path is not None and include_splits is not None:
            raise ValueError("Provide either manifest_path or include_splits, not both.")
        if manifest_path is None and include_splits is None:
            include_splits = ["train-clean-100", "train-clean-360", "train-other-500"]

        if isinstance(include_splits, str):
            include_splits = [include_splits]

        self.sampling_rate = sampling_rate
        self.num_audio_channels = num_audio_channels
        self.tokenizer = tokenizer
        self.train_split = train_split
        self.apply_spec_augment = apply_spec_augment
        self.apply_audio_augment = apply_audio_augment
        self.mode = mode
        self.spec_augment = ConformerSpecAugment()

        # ── Load utterance list ───────────────────────────────────────────
        if manifest_path is not None:
            with open(manifest_path) as f:
                records = json.load(f)
            self.librispeech_data = [(r["path"], r["transcript"]) for r in records]
        else:
            self.librispeech_data = []
            for split in include_splits:
                path_to_split = os.path.join(path_to_data_root, split)
                for speaker in sorted(os.listdir(path_to_split)):
                    path_to_speaker = os.path.join(path_to_split, speaker)
                    if not os.path.isdir(path_to_speaker):
                        continue
                    for section in sorted(os.listdir(path_to_speaker)):
                        path_to_section = os.path.join(path_to_speaker, section)
                        if not os.path.isdir(path_to_section):
                            continue
                        files = os.listdir(path_to_section)
                        txt = [p for p in files if p.endswith(".txt")]
                        if not txt:
                            continue
                        with open(os.path.join(path_to_section, txt[0])) as f:
                            for line in f:
                                parts = line.split()
                                uid = parts[0]
                                transcript = " ".join(parts[1:]).strip()
                                audio_path = os.path.join(path_to_section, uid + ".flac")
                                self.librispeech_data.append((audio_path, transcript))

        # ── Mel transforms (used only in mel mode) ────────────────────────
        self.audio2mels = T.MelSpectrogram(sample_rate=sampling_rate, n_mels=80)
        self.amp2db = T.AmplitudeToDB(top_db=80.0)

    def __len__(self):
        return len(self.librispeech_data)

    def __getitem__(self, idx):
        path_to_audio, transcript = self.librispeech_data[idx]
        audio, orig_sr = torchaudio.load(path_to_audio, normalize=True)
        if orig_sr != self.sampling_rate:
            audio = torchaudio.functional.resample(
                audio, orig_freq=orig_sr, new_freq=self.sampling_rate
            )

        uid = os.path.splitext(os.path.basename(path_to_audio))[0]
        tokenized = (
            torch.tensor(self.tokenizer.encode(transcript))
            if self.tokenizer is not None
            else transcript
        )

        # ── Waveform mode ────────────────────────────────────────────────
        if self.mode == "waveform":
            return {
                "input_values":   audio.squeeze(0),  # [T] float32
                "raw_audio":      audio,              # [1, T]
                "raw_transcript": transcript,
                "labels":         tokenized,
                "uid":            uid,
            }

        # ── Mel mode ─────────────────────────────────────────────────────
        mel = self.audio2mels(audio)
        mel = self.amp2db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        if self.train_split and self.apply_spec_augment:
            mel = self.spec_augment(mel)

        return {
            "input_values":   mel[0].T,        # [T, 80]
            "raw_audio":      audio,
            "raw_transcript": transcript,
            "labels":         tokenized,
            "uid":            uid,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sample usage
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
    dataset = LibriSpeechDataset(
        path_to_data_root="./",
        include_splits="train-clean-100",
        tokenizer=tokenizer,
    )
    sample = next(iter(dataset))
    plt.figure(figsize=(15, 5))
    plt.imshow(sample["input_values"].T)
    plt.axis("off")
    plt.gca().invert_yaxis()
    plt.show()
