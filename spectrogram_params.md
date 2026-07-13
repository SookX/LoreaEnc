# Spectrogram Parameters

Source: `CausalSpecUnit/data.py` — `LogMelExtractor` (the active SSL pipeline).

| Parameter | Value |
|---|---|
| Sample rate | 16,000 Hz |
| FFT size (`n_fft`) | 512 points |
| Window length | 400 samples (25 ms) |
| Hop length | 160 samples (10 ms) |
| Mel bins | 80 |
| Power | 2.0 (power spectrogram) |
| Amplitude scaling | `AmplitudeToDB(top_db=80.0)` |
| Pre-emphasis | 0.97 (`x[t] - 0.97 * x[t-1]`) |
| Padding | `center=True`, reflect |
| Normalization | Global CMVN (mean/std loaded from `cmvn.pt`) |
| Output shape | `[T, 80]` (time-first) |

## Note
`dataset/precompute_mels.py` uses `n_fft=400` with no explicit `win_length` and
per-sample normalization — it is an older/unused script. The SSL pipeline always
uses `LogMelExtractor` from `CausalSpecUnit/data.py`.
