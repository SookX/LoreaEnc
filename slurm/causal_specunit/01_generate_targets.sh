#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_targets
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_targets.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_targets.%j.err

set -euo pipefail

module purge
module load anaconda3

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets"
FIGURES_DIR="outputs/causal_specunit/figures"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

cd "${PROJECT_DIR}"
mkdir -p logs "${TARGETS_DIR}" "${FIGURES_DIR}"

echo "Job ${SLURM_JOB_ID} target generation starting at $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Python: $(which python)"
echo "Data root: ${DATA_ROOT}"

python -m CausalSpecUnit.generate_targets \
    --data-root "${DATA_ROOT}" \
    --output-dir "${TARGETS_DIR}" \
    --chunk-size 4 \
    --chunk-stride 4 \
    --pca-dim 64 \
    --k-coarse 100 \
    --k-fine 500 \
    --max-fit-chunks 1000000

python -m CausalSpecUnit.visualize_clusters \
    --targets-dir "${TARGETS_DIR}" \
    --output-dir "${FIGURES_DIR}" \
    --top-n 24

echo "Job ${SLURM_JOB_ID} target generation finished at $(date)"

