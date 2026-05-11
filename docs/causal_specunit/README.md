# CausalSpecUnit Research Plan

This folder contains the concrete plan for a from-scratch self-supervised
pretraining method for compact ASR using spectrograms and a lightweight
SqueezeNet-style encoder.

Core idea:

```text
from-scratch SSL pretraining on spectrograms -> fine-tune SqueezeNet for ASR
```

The method does not use pretrained teacher models such as HuBERT, WavLM,
wav2vec 2.0, or Whisper. Targets are derived only from the audio/spectrogram
data.

Main method name:

```text
SpecUnit
```

Full title candidate:

```text
SpecUnit: From-Scratch Spectrogram Unit Prediction for Compact ASR
```

Use `CausalSpecUnit` only for a future streaming/causal-attention variant. The
current copied SqueezeFormer encoder is bidirectional.

## Files

```text
method.md       Technical method, target generation, architecture, ablations
losses.md       InfoNCE vs MSE vs cross-entropy for this setup
pseudocode.md   Target-generation and pretraining pseudocode
```
