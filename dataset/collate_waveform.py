"""
Waveform collate function for HuggingFace wav2vec2 / HuBERT processors.
Returns padded raw waveforms with attention_mask.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional


def collate_fn_waveform(
    batch: List[Dict[str, Any]],
    feature_extractor=None,
    sampling_rate: int = 16_000,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    batch: list of dicts from LibriSpeechDataset (waveform mode).
      Each dict contains:
        - raw_audio  : Tensor[1, T]  (mono)
        - labels     : str or Tensor
        - raw_transcript: str
        - uid        : str

    Returns dict with keys expected by HuggingFace models:
      - input_values   : FloatTensor[B, T_max]   (normalised)
      - attention_mask : LongTensor[B, T_max]
      - labels         : variable (transcript strings or token ids)
    """
    # Sort longest → shortest (optional, helps some padding strategies)
    batch = sorted(batch, key=lambda x: x["raw_audio"].shape[-1], reverse=True)

    raw_audios: List[np.ndarray] = [
        sample["raw_audio"].squeeze(0).numpy() for sample in batch
    ]
    raw_transcripts = [sample["raw_transcript"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    uids = [sample["uid"] for sample in batch]

    if feature_extractor is not None:
        # Let the HF processor handle normalisation + padding
        processed = feature_extractor(
            raw_audios,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
            max_length=max_length,
            truncation=max_length is not None,
        )
        input_values = processed["input_values"]
        attention_mask = processed.get(
            "attention_mask",
            torch.ones(input_values.shape[:2], dtype=torch.long),
        )
    else:
        # Manual zero-padding
        lengths = [len(a) for a in raw_audios]
        T = max(lengths)
        input_values = torch.zeros(len(batch), T)
        attention_mask = torch.zeros(len(batch), T, dtype=torch.long)
        for i, (a, l) in enumerate(zip(raw_audios, lengths)):
            t = torch.from_numpy(a).float()
            input_values[i, :l] = t
            attention_mask[i, :l] = 1

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels,
        "raw_transcripts": raw_transcripts,
        "uids": uids,
    }


class WaveformCollator:
    """
    Picklable callable wrapper around collate_fn_waveform.
    Use this instead of a lambda when setting DataLoader collate_fn,
    so that it can be pickled by multiprocessing (required on Windows).
    """

    def __init__(self, feature_extractor=None, sampling_rate: int = 16_000,
                 max_length: Optional[int] = None):
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate
        self.max_length = max_length

    def __call__(self, batch):
        return collate_fn_waveform(
            batch,
            feature_extractor=self.feature_extractor,
            sampling_rate=self.sampling_rate,
            max_length=self.max_length,
        )
