#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_cache_mels
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_cache_mels.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_cache_mels.%j.err

set -euo pipefail

module purge
module load anaconda3

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h"
MEL_CACHE_DIR="outputs/causal_specunit/mel_cache_960h"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

cd "${PROJECT_DIR}"
mkdir -p logs "${MEL_CACHE_DIR}"

echo "Job ${SLURM_JOB_ID} mel caching starting at $(date)"
echo "Data root: ${DATA_ROOT}"
echo "Targets: ${TARGETS_DIR}"
echo "Mel cache: ${MEL_CACHE_DIR}"

python -m CausalSpecUnit.cache_mels \
    --data-root "${DATA_ROOT}" \
    --targets-dir "${TARGETS_DIR}" \
    --output-dir "${MEL_CACHE_DIR}" \
    --splits train-clean-100 train-clean-360 train-other-500

echo "Job ${SLURM_JOB_ID} mel caching finished at $(date)"
