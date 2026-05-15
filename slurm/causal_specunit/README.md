# SpecUnit Slurm Pipeline

Run these jobs in order:

```bash
sbatch slurm/causal_specunit/01_generate_targets.sh
TEXTGRID_DIR=/path/to/librispeech_textgrids sbatch slurm/causal_specunit/01b_phone_purity.sh
sbatch slurm/causal_specunit/02_pretrain_ssl_100k_c8.sh
sbatch slurm/run_baseline.sh
sbatch slurm/run_ssl_finetune.sh
```

Artifacts:

```text
outputs/causal_specunit/targets_960h_c8
outputs/causal_specunit/figures_960h_c8
outputs/causal_specunit/phone_purity_c8.npz
outputs/causal_specunit/phone_purity_c8_no_silence.npz
outputs/causal_specunit/pretrain_ssl_100k_c8/checkpoint_step100000/checkpoint.pt
outputs/squeezeformer_xs_150ep_scratch_cmvn_c8
outputs/squeezeformer_xs_150ep_ssl_cmvn_c8
```

The target job generates clean spectrogram k-means targets for the full 960h
LibriSpeech training set using 8-frame chunks with stride 4. The SSL job
pretrains for exactly 100,000 optimizer steps with HuBERT-style masked-unit
prediction from those targets. The baseline and SSL fine-tune jobs both use the
same CMVN frontend; the SSL fine-tune uses a lower encoder LR and a higher CTC
head LR instead of freezing the encoder.

For a quick validation run, use:

```bash
sbatch slurm/causal_specunit/02_pretrain_ssl_50k.sh
SSL_CHECKPOINT=outputs/causal_specunit/pretrain_ssl_50k_c8/checkpoint_step050000/checkpoint.pt \
OUTPUT_DIR=outputs/squeezeformer_xs_ssl_50k_c8_smoke \
sbatch slurm/run_ssl_finetune_smoke.sh
```

Note: this is not streaming-causal yet. The copied SqueezeFormer baseline uses
bidirectional attention. The "causal" folder name is historical; use `SpecUnit`
when describing the current method.
