#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_prep_librilight
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_prep_librilight.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_prep_librilight.%j.err

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
