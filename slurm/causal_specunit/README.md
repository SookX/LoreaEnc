# SpecUnit Slurm Pipeline

Run these jobs in order:

```bash
sbatch slurm/causal_specunit/01_generate_targets.sh
TEXTGRID_DIR=/path/to/librispeech_textgrids sbatch slurm/causal_specunit/01b_phone_purity.sh
sbatch slurm/causal_specunit/02_pretrain_ssl_50k.sh
sbatch slurm/causal_specunit/03_train_ctc_150ep.sh
```

Artifacts:

```text
outputs/causal_specunit/targets
outputs/causal_specunit/figures
outputs/causal_specunit/phone_purity.npz
outputs/causal_specunit/phone_purity_no_silence.npz
outputs/causal_specunit/pretrain_ssl_50k/checkpoint_step050000/checkpoint.pt
outputs/causal_specunit/ctc_ssl_150ep
```

The SSL job pretrains for exactly 50,000 optimizer steps with HuBERT-style
masked-unit prediction from clean spectrogram k-means targets. The CTC job then
fine-tunes the copied SqueezeFormer-XS encoder for 150 epochs using the SSL
checkpoint.

Note: this is not streaming-causal yet. The copied SqueezeFormer baseline uses
bidirectional attention. The "causal" folder name is historical; use `SpecUnit`
when describing the current method.
