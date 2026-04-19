import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2CTCTokenizer
import pandas as pd


class LibriSpeechDataset(Dataset):
    """
    LibriSpeechDataset for spectrogram-based pre-training.

    Returns mel spectrograms instead of raw waveforms. The spectrogram frames
    are the direct input to the transformer (no convolutional feature encoder),
    so masking and negative sampling operate in frame space here rather than
    being derived from conv strides as in wav2vec2.

    Splits:
        Training:   ["train-clean-100", "train-clean-360", "train-other-500"]
        Validation: ["dev-clean", "test-clean"]

    Requires audio_durations.csv in each section directory (same as wav2vec2).
    """

    def __init__(self,
                 path_to_data_root,
                 include_splits=["train-clean-100", "train-clean-360", "train-other-500"],
                 max_audio_duration=20.0,
                 min_audio_duration=2.0,
                 sampling_rate=16000,
                 # Mel spectrogram config
                 n_fft=400,
                 hop_length=160,         # 10ms hop at 16kHz -> 100 frames/sec
                 n_mels=80,
                 truncate_audio=True,
                 return_transcripts=False,
                 hf_model_name="facebook/wav2vec2-base"):

        if isinstance(include_splits, str):
            include_splits = [include_splits]

        self.sampling_rate      = sampling_rate
        self.return_transcripts = return_transcripts
        self.truncate_audio     = truncate_audio
        self.hop_length         = hop_length
        self.n_mels             = n_mels
        self.min_audio_samples  = int(min_audio_duration * sampling_rate)
        self.max_audio_samples  = int(max_audio_duration * sampling_rate)

        # Mel spectrogram extractor — runs in dataloader workers (CPU)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

        ### Scan dataset ###
        self.librispeech_data = []
        for split in include_splits:
            path_to_split = os.path.join(path_to_data_root, split)

            for speaker in os.listdir(path_to_split):
                path_to_speaker = os.path.join(path_to_split, speaker)

                for section in os.listdir(path_to_speaker):
                    path_to_section = os.path.join(path_to_speaker, section)

                    files = os.listdir(path_to_section)
                    transcript_file = [p for p in files if p.endswith(".txt")][0]

                    audio_durations = pd.read_csv(
                        os.path.join(path_to_section, "audio_durations.csv")
                    )
                    audio_durations_dict = audio_durations.set_index("root")["duration"].to_dict()

                    with open(os.path.join(path_to_section, transcript_file), "r") as f:
                        transcripts = f.readlines()

                    for line in transcripts:
                        parts = line.split()
                        audio_root = parts[0]
                        transcript  = " ".join(parts[1:]).strip()
                        duration    = audio_durations_dict[audio_root]
                        full_path   = os.path.join(path_to_section, audio_root + ".flac")

                        if (duration >= min_audio_duration) and (
                            duration <= max_audio_duration or truncate_audio
                        ):
                            self.librispeech_data.append((full_path, transcript))

        if return_transcripts:
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(hf_model_name)

    def __len__(self):
        return len(self.librispeech_data)

    def __getitem__(self, idx):
        path_to_audio, transcript = self.librispeech_data[idx]

        ### Load and optionally truncate ###
        audio, orig_sr = torchaudio.load(path_to_audio, num_frames=self.max_audio_samples)

        ### Resample if needed ###
        if orig_sr != self.sampling_rate:
            audio = torchaudio.functional.resample(
                audio, orig_freq=orig_sr, new_freq=self.sampling_rate
            )

        # audio: (1, T) — keep channel dim for MelSpectrogram which expects (..., T)
        # Output: (1, n_mels, T_frames)
        mel = self.mel_transform(audio)
        mel = self.amplitude_to_db(mel)

        # (n_mels, T_frames) — drop the channel dim
        mel = mel.squeeze(0)

        # Normalize to zero mean, unit variance per utterance
        mel = (mel - mel.mean()) / (mel.std() + 1e-7)

        # Transpose to (T_frames, n_mels) — time-first, matches transformer convention
        mel = mel.T

        sample = {"input_values": mel}

        if self.return_transcripts:
            sample["labels"] = torch.tensor(self.tokenizer.encode(transcript))

        return sample


class Spec2VecCollateFunctionForPreTraining:
    """
    Collate for spectrogram pre-training.

    Unlike wav2vec2, the spectrogram frames ARE the model input — there is no
    convolutional feature encoder to derive a sub-attention-mask from. The
    attention mask and span mask both operate directly in frame space.

    Masking spans are applied here (in the collate function) and stored as
    mask_time_indices so the model can replace masked frames with a learned
    mask embedding before passing to the transformer.
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, batch):
        from utils import compute_span_mask, sample_negative_indices

        config = self.config
        mels   = [sample["input_values"] for sample in batch]  # each (T_i, n_mels)

        # Attention mask: 1 for real frames, 0 for padding
        attention_mask = [torch.ones(m.shape[0], dtype=torch.float) for m in mels]

        # Pad to longest sequence in batch along time axis
        # pad_sequence expects (T, ...) tensors — mels are already (T, n_mels)
        padded_mels    = torch.nn.utils.rnn.pad_sequence(mels,           batch_first=True, padding_value=0.0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0.0)

        # (B, T_max, n_mels)
        batch_size, seq_len, _ = padded_mels.shape

        # Span mask: same interface as wav2vec2 — (B, T_max) bool
        span_mask = compute_span_mask(
            shape=(batch_size, seq_len),
            mask_prob=config.masking_probability,
            mask_length=config.masking_span_length,
            min_masks=config.minimum_spans,
            attention_mask=attention_mask.bool(),
        )

        # Negative indices for contrastive loss: (B, T_max, num_negatives)
        sampled_negatives = sample_negative_indices(
            features_shape=(batch_size, seq_len),
            num_negatives=config.num_negatives,
            mask_time_indices=span_mask,
        )

        return {
            "input_values":           padded_mels,           # (B, T, n_mels)
            "attention_mask":         attention_mask.bool(),  # (B, T)
            "mask_time_indices":      span_mask,              # (B, T)
            "sampled_negative_indices": sampled_negatives,    # (B, T, num_neg)
        }


def plot_spectrogram(mel, hop_length=160, sampling_rate=16000, title=None, save_path=None):
    """
    Plot a single mel spectrogram.

    Args:
        mel:           Tensor or ndarray of shape (T_frames, n_mels) — time-first,
                       as returned by LibriSpeechDataset.__getitem__.
        hop_length:    Hop length used during extraction (for time axis labelling).
        sampling_rate: Sample rate (for time axis labelling).
        title:         Optional plot title.
        save_path:     If provided, saves the figure to this path instead of showing it.
    """
    import matplotlib.pyplot as plt

    if hasattr(mel, "numpy"):
        mel = mel.numpy()

    # mel is (T, n_mels) — transpose to (n_mels, T) for imshow
    mel = mel.T

    n_mels, n_frames = mel.shape
    duration = n_frames * hop_length / sampling_rate

    fig, ax = plt.subplots(figsize=(12, 4))
    img = ax.imshow(
        mel,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        extent=[0, duration, 0, n_mels],
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title(title or "Mel Spectrogram")
    plt.colorbar(img, ax=ax, format="%+.1f dB")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_spectrogram_with_mask(mel, mask, hop_length=160, sampling_rate=16000,
                                title=None, save_path=None):
    """
    Plot a mel spectrogram with masked spans highlighted in red.

    Args:
        mel:   Tensor or ndarray of shape (T_frames, n_mels).
        mask:  Bool tensor or ndarray of shape (T_frames,) — True = masked frame.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if hasattr(mel,  "numpy"): mel  = mel.numpy()
    if hasattr(mask, "numpy"): mask = mask.numpy()

    mel = mel.T  # (n_mels, T)
    n_mels, n_frames = mel.shape
    duration = n_frames * hop_length / sampling_rate
    frame_width = hop_length / sampling_rate

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(
        mel,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        extent=[0, duration, 0, n_mels],
    )

    # Highlight masked frames
    for t, masked in enumerate(mask):
        if masked:
            ax.add_patch(Rectangle(
                (t * frame_width, 0), frame_width, n_mels,
                linewidth=0, edgecolor="none", facecolor="red", alpha=0.4,
            ))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title(title or "Mel Spectrogram with Span Mask")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from utils import Spec2VecConfig

    dataset = LibriSpeechDataset("./dataset", include_splits=["dev-clean"])
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Mel shape: {sample['input_values'].shape}")  # (T_frames, 80)

    config = Spec2VecConfig()
    loader = DataLoader(
        dataset, batch_size=4,
        collate_fn=Spec2VecCollateFunctionForPreTraining(config)
    )
    batch = next(iter(loader))
    for k, v in batch.items():
        print(f"{k}: {v.shape}")
