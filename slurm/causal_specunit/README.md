# SpecUnit Slurm Pipeline

Run these jobs in order:

```bash
sbatch slurm/causal_specunit/01_generate_targets.sh
TEXTGRID_DIR=/path/to/librispeech_textgrids sbatch slurm/causal_specunit/01b_phone_purity.sh
sbatch slurm/causal_specunit/02_pretrain_ssl_150k.sh
sbatch slurm/causal_specunit/03_train_ctc_150ep.sh
```

Artifacts:

```text
outputs/causal_specunit/targets_960h
outputs/causal_specunit/figures_960h
outputs/causal_specunit/phone_purity.npz
outputs/causal_specunit/phone_purity_no_silence.npz
outputs/causal_specunit/pretrain_ssl_150k/checkpoint_step150000/checkpoint.pt
outputs/causal_specunit/ctc_ssl_150ep
```

The target job generates clean spectrogram k-means targets for the full 960h
LibriSpeech training set. The SSL job pretrains for exactly 150,000 optimizer
steps with HuBERT-style masked-unit prediction from those targets. The CTC job then
fine-tunes the copied SqueezeFormer-XS encoder for 150 epochs using the SSL
checkpoint.

Note: this is not streaming-causal yet. The copied SqueezeFormer baseline uses
bidirectional attention. The "causal" folder name is historical; use `SpecUnit`
when describing the current method.
