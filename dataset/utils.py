import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.signal import resample
from scipy.signal import hilbert
import emd

def plot_audio(waveform):
    plt.plot(audio)
    plt.show();

def plot_imfs(imf):
    n_samples, n_imfs = imf.shape

    torchaudio.save("./imf1.wav", torch.asarray(imf[:, 1], dtype=torch.float32).unsqueeze(0), sampling_rate)
    fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(12, 2*(n_imfs+1)), sharex=True)

    axes[0].plot(audio.numpy())
    axes[0].set_title("Original Signal")

    for i in range(n_imfs):
        axes[i+1].plot(imf[:, i])
        axes[i+1].set_title(f"IMF {i+1}")

    plt.xlabel("Samples")
    plt.tight_layout()
    plt.show()

def plot_audio_vs_reconstruction(audio, imf):
    reconstructed_audio = torch.asarray(imf.sum(axis = 1), dtype=torch.float32)
    plt.figure(figsize=(12, 6))
    plt.plot(audio.numpy(), label="Original", alpha=0.7)
    plt.plot(reconstructed_audio.numpy(), label="Reconstructed (sum of IMFs)", alpha=0.7, linestyle="--")
    plt.legend()
    plt.title("Original vs Reconstructed Audio")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()

def hilbert_huang_spectrogram(audio, imf, sample_rate, n_freq=80, n_time=1024, overlap=0.5):
    """
    Compute and plot Hilbert-Huang Spectrum (HHS) from IMFs.
    """

    n_samples, n_imfs = imf.shape
    hop = int((1 - overlap) * n_samples / n_time)
    win = int(n_samples / n_time)

    hhs = np.zeros((n_freq, n_time))

    for t_idx, start in enumerate(range(0, n_samples - win, hop)):
        if t_idx >= n_time: 
            break

        segment = imf[start:start+win, :]

        for i in range(n_imfs):
            analytic = hilbert(segment[:, i])
            amp = np.abs(analytic)
            phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(phase) * sample_rate / (2.0 * np.pi)

            amp = amp[:-1]
            inst_freq = np.clip(inst_freq, 0, sample_rate/2) 

            f_bins = np.round(inst_freq / (sample_rate/2) * (n_freq-1)).astype(int)
            f_bins = np.clip(f_bins, 0, n_freq-1)

            for f, a in zip(f_bins, amp):
                hhs[f, t_idx] += a


    return hhs