"""
dataset/dataset.py
──────────────────
LibriSpeechDataset — loads precomputed mel spectrograms directly from NFS.
Each .flac file has a corresponding .mel.pt file (fp16, shape [T, 80])
sitting alongside it, produced by dataset/precompute_mels.py.

No RAM preloading. No waveform decoding. Just torch.load() per sample.
"""
import os
import json
import torch
import torchaudio
import soundfile as _sf
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
from typing import List, Optional
from transformers import Wav2Vec2CTCTokenizer


class ConformerSpecAugment(torch.nn.Module):
    def __init__(self, time_mask_param=100, freq_mask_param=27,
                 num_time_masks=5, num_freq_masks=2):
        super().__init__()
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.num_time_masks  = num_time_masks
        self.num_freq_masks  = num_freq_masks

    def forward(self, spec):          # spec: [B, 80, T]
        out = spec.clone()
        for _ in range(self.num_freq_masks):
            out = self.freq_mask(out)
        for _ in range(self.num_time_masks):
            out = self.time_mask(out)
        return out


class LibriSpeechDataset(Dataset):
    """
    Loads pre-computed mel spectrograms from .mel.pt files on NFS.
    Falls back to soundfile waveform loading if .mel.pt is missing.

    self.lengths  — list of mel frame counts (T), used by the bucketed sampler.
    """

    def __init__(
        self,
        path_to_data_root: str,
        include_splits: List[str] = None,
        manifest_path: Optional[str] = None,
        sampling_rate: int = 16_000,
        tokenizer=None,
        hop_length: int = 160,
        scan_lengths: bool = True,
    ):
        if manifest_path is not None and include_splits is not None:
            raise ValueError("Provide either manifest_path or include_splits, not both.")
        if manifest_path is None and include_splits is None:
            include_splits = ["train-clean-100", "train-clean-360", "train-other-500"]
        if isinstance(include_splits, str):
            include_splits = [include_splits]

        self.sampling_rate = sampling_rate
        self.tokenizer     = tokenizer
        self.hop_length    = hop_length

        # ── Build utterance list ─────────────────────────────────────────
        if manifest_path is not None:
            with open(manifest_path) as f:
                records = json.load(f)
            self.librispeech_data = [(r["path"], r["transcript"]) for r in records]
        else:
            self.librispeech_data = []
            for split in include_splits:
                split_dir = os.path.join(path_to_data_root, split)
                for speaker in sorted(os.listdir(split_dir)):
                    spk_dir = os.path.join(split_dir, speaker)
                    if not os.path.isdir(spk_dir):
                        continue
                    for section in sorted(os.listdir(spk_dir)):
                        sec_dir = os.path.join(spk_dir, section)
                        if not os.path.isdir(sec_dir):
                            continue
                        txts = [p for p in os.listdir(sec_dir) if p.endswith(".txt")]
                        if not txts:
                            continue
                        with open(os.path.join(sec_dir, txts[0])) as f:
                            for line in f:
                                parts = line.split()
                                uid   = parts[0]
                                text  = " ".join(parts[1:]).strip()
                                self.librispeech_data.append(
                                    (os.path.join(sec_dir, uid + ".flac"), text)
                                )

        # ── Scan audio lengths for bucketed sampler (reads FLAC headers only) ─
        if scan_lengths:
            def _dur(path_text):
                path, _ = path_text
                try:
                    return _sf.info(path).frames // hop_length
                except Exception:
                    return 400  # ~8 s fallback
            with ThreadPoolExecutor(max_workers=16) as pool:
                self.lengths = list(pool.map(_dur, self.librispeech_data))
        else:
            self.lengths = [400] * len(self.librispeech_data)

    def __len__(self):
        return len(self.librispeech_data)

    def __getitem__(self, idx):
        path, transcript = self.librispeech_data[idx]
        mel_path = path.replace(".flac", ".mel.pt")

        # ── Primary: load precomputed mel (fp16, [T, 80]) ────────────────
        try:
            mel = torch.load(mel_path, weights_only=True)  # fp16 [T, 80]
        except FileNotFoundError:
            # ── Fallback: decode FLAC on the fly ─────────────────────────
            audio_np, sr = _sf.read(path, dtype="float32", always_2d=False)
            audio = torch.from_numpy(audio_np)
            if sr != self.sampling_rate:
                import torchaudio
                audio = torchaudio.functional.resample(
                    audio.unsqueeze(0), sr, self.sampling_rate
                ).squeeze(0)
            # Return waveform; training loop handles it
            mel = audio.unsqueeze(0)   # [1, T_wav] — dim check tells us it's audio

        labels = (
            torch.tensor(self.tokenizer.encode(transcript))
            if self.tokenizer is not None else transcript
        )
        return {"input_values": mel, "raw_transcript": transcript, "labels": labels}
