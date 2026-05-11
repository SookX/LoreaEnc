# Spectrogram SSL Plan

## Main Idea

The core contribution is spectrogram-native semi-supervised learning for CTC ASR.
Instead of learning SSL representations from waveforms, the model learns directly
from log-mel spectrograms and then fine-tunes with a CTC head.

The broader motivation is compact, low-resource, edge-oriented ASR. SqueezeFormer
is used as the baseline because it is an efficient ASR architecture and fits the
goal of building a smaller model that can benefit from unlabeled audio without
depending on a large waveform SSL encoder.

Research hypothesis:

```text
Spectrogram-domain SSL can learn useful acoustic representations without waveform
pretraining, and these representations transfer effectively to CTC-based ASR.
```

Edge-ASR framing:

```text
Large waveform SSL models work well, but they are expensive. For low-resource or
edge ASR, we want a compact model trained from spectrograms, using unlabeled
audio to improve performance without relying on giant waveform encoders.
```

Paper goal:

```text
We investigate whether discrete self-supervised targets derived directly from
log-mel spectrograms can improve compact SqueezeFormer CTC ASR models,
especially in low-label regimes relevant to edge deployment.
```

Possible titles:

```text
Spectrogram-Derived Discrete Pretraining for Compact CTC ASR
Spectrogram-Native Self-Supervision for Low-Resource Edge ASR
```

## Recommended Direction

Use spectrogram discrete-token prediction rather than ViT-style patching as the
main method.

Pipeline:

```text
audio
-> log-mel spectrogram [T, 80]
-> derive k-means targets from spectrogram frames or short frame windows
-> mask spans/regions of the spectrogram
-> SqueezeFormer predicts discrete k-means units at masked time steps
-> initialize CTC model from SSL encoder
-> fine-tune on labeled LibriSpeech
```

This keeps the time axis intact, which is important for CTC alignment.

## Target Options

Frame-level targets:

```text
one k-means label per spectrogram frame
```

Pros: simplest and naturally CTC-compatible.  
Cons: can be noisy.

Stacked-frame targets:

```text
target at time t = k-means(log-mel[t-2:t+2, :])
```

Pros: still time-preserving, less noisy, captures local context.  
Cons: slightly more preprocessing.

2D patch targets:

```text
k-means over time-frequency spectrogram patches
```

Pros: closest to the original ViT-style idea.  
Cons: more complex and riskier for CTC because time structure must be recovered.

Recommended first version:

```text
stacked-frame spectrogram units
```

## SSL Objective

Best first objective:

```text
masked discrete unit prediction
```

Training:

```text
input: masked log-mel spectrogram
target: k-means unit IDs for masked positions
loss: cross entropy over cluster IDs, computed mainly/only on masked positions
```

Alternative baseline objective:

```text
masked spectrogram reconstruction
```

This is easier but may learn less semantic/acoustic structure than discrete
unit prediction.

## Patching Decision

Patching is not required for the main method.

For CTC ASR, the safest design is:

```text
spectrogram -> encoder -> [B, T', D] -> CTC
```

If patching is tested, it should preserve or recover a time sequence before CTC:

```text
mel [B, T, 80]
-> small time-frequency patches
-> encoder
-> pool over frequency patches
-> [B, T_patch, D]
-> CTC
```

Naive AST-style global classification with a class token is not appropriate for
CTC ASR.

## Four-Page Paper Ablations

Keep the ablations focused. The main claim should be:

```text
Spectrogram-derived discrete SSL targets improve CTC ASR fine-tuning.
```

For the edge-ASR framing, emphasize:

```text
model size
training/fine-tuning data efficiency
WER improvement over scratch
compute-friendly spectrogram pipeline
```

Most important experiments:

1. Scratch baseline

```text
SqueezeFormer CTC from scratch
```

2. Main method

```text
spectrogram k-means SSL pretrain -> CTC fine-tune
```

3. Target type

```text
masked spectrogram reconstruction
vs
spectrogram k-means unit prediction
```

4. Target construction

```text
single-frame k-means targets
vs
stacked-frame k-means targets
```

5. Label fraction

```text
10h, 100h, full
```

If there is enough compute, also test:

```text
1h, 10h, 100h, full
```

6. Model size

```text
SqueezeFormer-XS vs SqueezeFormer-SM
```

This helps support the compact/edge-oriented story.

Optional ablations:

```text
K = 100, 500, 1000 clusters
time-span masking vs time-frequency block masking
frame/stacked-frame units vs 2D patch units
```

## Suggested Tables

Main ASR table:

```text
Method                         10h WER   100h WER   Full WER
CTC from scratch
Masked spectrogram recon + CTC
Spectrogram k-means SSL + CTC
```

Compact edge-ASR table:

```text
Model      Params   Input      Pretrain type       10h WER   100h WER
XS         9M       log-mel    none
XS         9M       log-mel    recon SSL
XS         9M       log-mel    discrete SSL
SM         ...      log-mel    none
SM         ...      log-mel    discrete SSL
```

Ablation table:

```text
Ablation                       dev-clean WER   dev-other WER
Time-span masking
Time-frequency masking
K=100
K=500
K=1000
Frame targets
Stacked-frame targets
```

If space is tight, prioritize:

```text
1. scratch vs SSL
2. reconstruction vs discrete targets
3. frame vs stacked-frame targets
4. label fraction: 10h vs 100h vs full
```

If reporting edge relevance is easy, include:

```text
parameter count
real-time factor or inference speed
CPU/GPU memory footprint
```

These are helpful but not mandatory for a 4-page workshop paper.
