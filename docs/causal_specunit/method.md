# SpecUnit Method

## Goal

The goal is to improve a compact SqueezeNet ASR baseline using from-scratch
self-supervised pretraining on spectrograms.

Pipeline:

```text
audio
-> 80-bin log-mel spectrogram
-> global per-frequency CMVN
-> extract full-band temporal chunks
-> flatten chunks
-> PCA compression
-> k-means clustering
-> discrete target sequence
-> causal SSL pretraining
-> CTC ASR fine-tuning
```

The main contribution is not waveform SSL and not teacher distillation. It is a
spectrogram-native, from-scratch SSL target-generation and prediction pipeline
for low-resource, edge-oriented ASR.

## Refined Idea

Learn discrete acoustic units from CMVN-normalized log-mel chunks using PCA and
hierarchical k-means. Then train a causal lightweight CNN to predict future
units from past spectrogram context. Finally, fine-tune the pretrained encoder
with a CTC head.

In compact form:

```text
masked spectrogram context -> clean spectrogram-derived unit labels -> CTC ASR
```

For each target time step:

```text
z_t^100 = coarse k-means label
z_t^500 = fine k-means label
```

The SSL model predicts clean targets at masked positions:

```text
SqueezeFormer(masked_x) -> z_t^100, z_t^500
```

Loss:

```text
L = CE(pred100[masked_positions], z100[masked_positions])
  + CE(pred500[masked_positions], z500[masked_positions])
```

The targets are generated from the clean spectrogram. Masking/SpecAugment is
applied only to the model input.

## Target Generation

Recommended first recipe:

```text
audio sample rate: 16 kHz
log-mel bins: 80
window: 25 ms
hop: 10 ms
normalization: global CMVN per mel bin
chunk size: 4 frames x 80 mel bins
chunk stride: 2 frames
flattened dim: 320
PCA dim: 64
PCA whitening: yes
clusters: K=100 and K=500
```

Why this recipe:

```text
4-frame chunks preserve phonetic transitions better than larger chunks.
stride 2 keeps useful temporal resolution.
PCA-64 denoises and makes k-means easier.
K=100 gives coarse units; K=500 gives fine units.
```

Use independent k-means models first:

```text
z100_t = kmeans100(PCA(chunk_t))
z500_t = kmeans500(PCA(chunk_t))
```

True nested hierarchical clustering can be tested later, but it is not required
for the first version.

## SqueezeNet Adaptation

Classic SqueezeNet is an image classifier and collapses spatial dimensions too
aggressively. For ASR, adapt it into a time-preserving spectrogram encoder.

Input:

```text
[B, 1, T, 80]
```

Output:

```text
[B, T', hidden]
```

SSL heads:

```text
head100: [B, T', hidden] -> [B, T', 100]
head500: [B, T', hidden] -> [B, T', 500]
```

Architecture constraints:

```text
Do not use global pooling over time.
Collapse frequency more aggressively than time.
Use causal padding along time.
Symmetric padding along frequency is acceptable.
Keep T' close to the target sequence length.
```

Suggested downsampling:

```text
time stride: 2, at most 4
frequency stride: 4 to 8 total
frequency pooling: yes
time pooling: minimal
```

Shape target:

```text
input T frames
target stride 2 frames
encoder output T' ~= T / 2
```

This keeps SSL targets and later CTC outputs easy to align.

## SSL Hyperparameters

Target generation:

```text
k-means samples: 1M-5M chunks
k-means init: k-means++
k-means iterations: 100-300
PCA dim: 64 first, 128 as ablation
chunk size: 4x80 first, 8x80 as ablation
clusters: K=100 and K=500
```

Pretraining:

```text
optimizer: AdamW
learning rate: 3e-4 to 1e-3
weight decay: 1e-4 to 5e-4
dropout: 0.1
prediction delay: d=2 first, d=4 as ablation
loss: CE_100 + CE_500
epochs: 50-200 depending data
```

Fine-tuning:

```text
initialize encoder from SSL checkpoint
replace SSL heads with CTC head
fine-tune all layers
use lower LR than scratch
compare against same SqueezeNet architecture trained from scratch
```

## Most Important Ablations

Priority ablations:

```text
1. Scratch ASR vs SSL-pretrained ASR
2. Raw spectrogram k-means vs PCA-compressed chunk k-means
3. K=100 only vs K=500 only vs K=100+500
4. Chunk size: 4x80 vs 8x80
5. Delay: d=1 vs d=2 vs d=4
6. Causal predictive SSL vs non-causal masked SSL
7. Reconstruction SSL vs discrete prediction SSL
8. Label fraction: 1h / 10h / 100h
```

The most important result:

```text
SSL should help most in low-label ASR fine-tuning.
```

## Failure Modes

Targets are too low-level:

```text
Symptom: SSL loss improves, but ASR WER does not.
Fix: increase delay d, use 8-frame chunks, use K=500, add stronger augmentation.
```

Targets are too noisy:

```text
Symptom: SSL loss stays high, cluster usage is uneven.
Fix: use PCA whitening, reduce K, use more k-means samples, check CMVN.
```

Time resolution is too low:

```text
Symptom: CTC fine-tuning is poor.
Fix: reduce chunk stride, reduce encoder time downsampling, use stride 2 not 4.
```

Model cheats with local continuity:

```text
Symptom: d=1 SSL is easy but transfers weakly.
Fix: use d=2 or d=4, mask/drop current local region, add temporal jitter.
```

Cluster collapse:

```text
Symptom: a few clusters dominate.
Fix: inspect histogram, reduce K, use PCA whitening, improve sampling.
```

Causal model underperforms:

```text
Symptom: causal SSL learns less than non-causal SSL.
Fix: allow small right context during ASR fine-tuning, e.g. 2-4 frames.
```

## Paper Framing

Main claim:

```text
Simple spectrogram-derived discrete units provide useful SSL supervision for
lightweight ASR models, especially in low-label regimes.
```

Pitch:

```text
We propose a from-scratch spectrogram-domain SSL method for compact ASR. Unlike
waveform SSL systems that rely on large pretrained teachers, our method derives
discrete predictive targets directly from log-mel spectrogram chunks using PCA
and hierarchical k-means. A causal SqueezeNet-style encoder is pretrained to
predict future acoustic units and then fine-tuned with CTC.
```

Do not claim SOTA unless results justify it. Claim compactness, simplicity,
from-scratch training, and low-resource usefulness.

## Comparisons

Raw spectrogram k-means:

```text
Cluster individual 80-dim frames directly.
Likely too local/noisy, but important baseline.
```

PCA chunk k-means:

```text
Uses short temporal context and denoising/compression.
Expected to produce better targets.
```

Reconstruction SSL:

```text
Predict masked log-mel values.
Easy baseline, but may overfocus on local spectrogram texture.
```

Non-causal masked SSL:

```text
Bidirectional context predicts masked units.
Likely stronger representation, but less streaming/edge-friendly.
```

Causal future-unit prediction:

```text
Harder and more aligned with streaming ASR.
Best fit for edge/causal deployment story.
```

Recommended first complete experiment:

```text
Target: 4x80 chunks, stride 2, PCA-64, K=100+500
SSL: causal SqueezeNet, d=2
Fine-tune: CTC on 10h and 100h
Compare: scratch, reconstruction SSL, raw-frame k-means SSL
```
