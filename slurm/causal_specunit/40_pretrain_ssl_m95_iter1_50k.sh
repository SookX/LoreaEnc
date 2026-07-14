#!/bin/bash
# Short SqueezeFormer-M95 iter-1 SSL probe with the safer large-model recipe.
#
# This is intentionally a diagnostic run, not the matched 400k wav2vec2-Base
# budget. It writes to a separate output directory and produces
# checkpoint_step050000 for downstream 1h fine-tune testing.
#
# Submit:
#   sbatch slurm/causal_specunit/40_pretrain_ssl_m95_iter1_50k.sh
#
# Optional knobs:
#   LR=3e-4 WARMUP_STEPS=10000 sbatch ...
#   BATCH_SIZE=128 GRAD_ACCUM_STEPS=1 sbatch ...

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=m95_i1_50k
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/m95_i1_50k.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/m95_i1_50k.%j.err

set -euo pipefail

export MAX_STEPS="${MAX_STEPS:-50000}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/ssl_m95_iter1_50k}"
export LR="${LR:-5e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-10000}"
export PEAK_STEPS="${PEAK_STEPS:-5000}"

exec bash slurm/causal_specunit/40_pretrain_ssl_m95_iter1_400k.sh
