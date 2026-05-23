#!/bin/bash
#SBATCH --job-name=prep_librilight
#SBATCH --output=logs/prep_librilight.%j.out
#SBATCH --error=logs/prep_librilight.%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
RAW_ROOT="${RAW_ROOT:-dataset/datasets/librilight}"
MANIFEST_ROOT="${MANIFEST_ROOT:-dataset/manifests/librilight}"
STATS_PATH="${STATS_PATH:-dataset/manifests/librilight/stats.json}"

mkdir -p logs

python dataset/prepare_librilight_limited.py \
  --data-root "${DATA_ROOT}" \
  --raw-root "${RAW_ROOT}" \
  --manifest-root "${MANIFEST_ROOT}" \
  --stats-path "${STATS_PATH}" \
  --smoke-test
