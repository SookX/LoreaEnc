# SpecUnit

From-scratch spectrogram SSL for compact ASR using a copied SqueezeFormer
baseline encoder.

This folder is separate from the root `SqueezeFormer/` baseline. The baseline
architecture files are copied into `CausalSpecUnit/squeezeformer_baseline/` so
experiments can be changed here without deleting, replacing, or modifying the
original baseline code.

## Naming

```text
SpecUnit
```

The current implementation is HuBERT-style masked-unit prediction with a copied
SqueezeFormer encoder. It is not a streaming/causal ASR model yet: copied
SqueezeFormer uses bidirectional attention. If we later add attention masks or
limited right context, we can use the name `CausalSpecUnit` for that variant.

## Method

```text
audio
-> 80-bin log-mel spectrogram
-> global per-frequency CMVN
-> full-band temporal chunks
-> PCA compression
-> K=100 and K=500 k-means labels
-> HuBERT-style masked-unit SSL pretraining with the copied SqueezeFormer encoder
-> CTC ASR fine-tuning
```

The SSL objective is:

```text
clean spectrogram -> clean z100/z500 targets
clean spectrogram -> time masking / SpecAugment-style corruption -> model input
SqueezeFormer(corrupted spectrogram) -> pred100, pred500
L = CE(pred100[masked_positions], z100[masked_positions])
  + CE(pred500[masked_positions], z500[masked_positions])
```

This keeps the system teacher-free: no HuBERT, WavLM, wav2vec 2.0, Whisper, or
other pretrained target model is used. The setup is HuBERT-style in objective,
but the targets are discovered from clean spectrogram chunks.

## Files

```text
common.py             DDP, checkpoint, scheduler, length helpers
data.py               LibriSpeech walking, log-mel extraction, collates
model.py              Wrappers around copied SqueezeFormer encoder/CTC model
squeezeformer_baseline/ Copied SqueezeFormer architecture files
generate_targets.py   Clean CMVN, PCA, k-means, target assignment
pretrain_ssl.py       HuBERT-style masked-unit SSL pretraining
train_ctc.py          CTC fine-tuning with baseline-like DDP training
visualize_clusters.py cluster usage, centroids, PCA, t-SNE, timelines, transitions
evaluate_phone_purity.py cluster-vs-phone purity/NMI using MFA TextGrids
```

## 1. Generate Targets

Small smoke version:

```bash
python -m CausalSpecUnit.generate_targets \
  --data-root dataset/datasets/librispeech/LibriSpeech \
  --output-dir outputs/causal_specunit/targets_smoke \
  --max-utterances 200 \
  --max-fit-chunks 50000
```

Full version:

```bash
python -m CausalSpecUnit.generate_targets \
  --data-root dataset/datasets/librispeech/LibriSpeech \
  --output-dir outputs/causal_specunit/targets \
  --chunk-size 4 \
  --chunk-stride 4 \
  --pca-dim 64 \
  --k-coarse 100 \
  --k-fine 500 \
  --max-fit-chunks 1000000
```

Outputs:

```text
cmvn.pt
cluster_artifacts.joblib
cluster_stats.npz
metadata.json
targets.pt
```

## 2. Visualize Clusters

```bash
python -m CausalSpecUnit.visualize_clusters \
  --targets-dir outputs/causal_specunit/targets \
  --output-dir outputs/causal_specunit/figures
```

Generated figures:

```text
pca_explained_variance.png
k100_usage.png
k500_usage.png
k100_coverage.png
k500_coverage.png
k100_centroids.png
k500_centroids.png
k100_tsne.png
k500_tsne.png
k100_transition_matrix.png
k500_transition_matrix.png
k100_label_timelines.png
k500_label_timelines.png
cluster_visualization_explanation.md
```

What they explain:

```text
PCA variance: how compressible full-band temporal chunks are.
Usage histograms: whether k-means collapsed into a few dominant units.
Coverage curves: how many clusters explain most of the target stream.
Centroid grids: prototype acoustic time-frequency patterns discovered from scratch.
t-SNE plots: geometry and frequency of the learned unit inventory.
Transition matrices: local temporal structure in the unit stream.
Label timelines: whether example utterances have varied, non-collapsed labels.
```

## 3. SSL Pretraining

## Optional: Phone Purity Diagnostic

If you have MFA or Zenodo LibriSpeech TextGrid alignments, check whether the
from-scratch units align with phones before committing to a long SSL run:

```bash
python -m CausalSpecUnit.evaluate_phone_purity \
  --targets-dir outputs/causal_specunit/targets \
  --textgrid-dir /path/to/librispeech_textgrids \
  --tier phones \
  --output outputs/causal_specunit/phone_purity.npz
```

To ignore silence/noise intervals:

```bash
python -m CausalSpecUnit.evaluate_phone_purity \
  --targets-dir outputs/causal_specunit/targets \
  --textgrid-dir /path/to/librispeech_textgrids \
  --tier phones \
  --exclude-silence
```

Useful rough interpretation:

```text
random purity over ~40 phones: ~2-5%
below ~20%: targets may be too noisy
40-60%: strong for first-pass discovered acoustic units
K=500 > K=100: good sign that fine units track finer phone distinctions
```

## 3. SSL Pretraining

Single GPU:

```bash
python -m CausalSpecUnit.pretrain_ssl \
  --data-root dataset/datasets/librispeech/LibriSpeech \
  --targets-dir outputs/causal_specunit/targets \
  --output-dir outputs/causal_specunit/pretrain \
  --batch-size 128 \
  --variant xs \
  --mask-prob 0.065 \
  --mask-length 10
```

Two GPUs:

```bash
torchrun --nproc_per_node=2 -m CausalSpecUnit.pretrain_ssl \
  --data-root dataset/datasets/librispeech/LibriSpeech \
  --targets-dir outputs/causal_specunit/targets \
  --output-dir outputs/causal_specunit/pretrain \
  --batch-size 128 \
  --variant xs \
  --mask-prob 0.065 \
  --mask-length 10
```

## 4. CTC Fine-Tuning

From scratch:

```bash
torchrun --nproc_per_node=2 -m CausalSpecUnit.train_ctc \
  --data-root dataset/datasets/librispeech/LibriSpeech \
  --cmvn-path outputs/causal_specunit/targets/cmvn.pt \
  --output-dir outputs/causal_specunit/ctc_scratch \
  --tokenizer-path dataset/bpe128.model \
  --variant xs \
  --batch-size 128 \
  --grad-accum-steps 2
```

From SSL checkpoint:

```bash
torchrun --nproc_per_node=2 -m CausalSpecUnit.train_ctc \
  --data-root dataset/datasets/librispeech/LibriSpeech \
  --cmvn-path outputs/causal_specunit/targets/cmvn.pt \
  --ssl-checkpoint outputs/causal_specunit/pretrain/checkpoint_ep100/checkpoint.pt \
  --output-dir outputs/causal_specunit/ctc_ssl \
  --tokenizer-path dataset/bpe128.model \
  --variant xs \
  --batch-size 128 \
  --grad-accum-steps 2
```

The CTC training intentionally follows the baseline structure:

```text
DDP with torchrun
AdamW
extended Noam schedule
bf16 autocast on CUDA
gradient clipping
rank-0 tqdm progress
checkpoint_best and periodic checkpoints
```

## First Recommended Experiment

```text
Targets: 4x80 chunks, stride 4, PCA-64, K=100+500
SSL: copied SqueezeFormer-XS, HuBERT-style masked-unit prediction
Fine-tune: CTC on 10h and 100h
Compare:
  1. copied SqueezeFormer CTC from scratch
  2. CausalSpecUnit SSL -> CTC
  3. raw-frame k-means SSL -> CTC
  4. reconstruction SSL -> CTC
```
