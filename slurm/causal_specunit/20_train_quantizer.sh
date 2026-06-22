#!/bin/bash
# Train one learned vector quantizer (VQ / RVQ / ParallelVQ) on the
# PCA-64 chunks that the k-means baseline uses. Saves state.pt +
# metrics.json + train.jsonl into OUTPUT_DIR.
#
# First run builds a chunks cache (~22 GB at
# source-targets-dir/pca64_chunks.npy), which subsequent runs reuse.
#
# Submit (one cell at a time):
#   QUANTIZER_TYPE=vq K1=600 sbatch slurm/causal_specunit/20_train_quantizer.sh
#   QUANTIZER_TYPE=rvq K1=100 K2=500 sbatch slurm/causal_specunit/20_train_quantizer.sh
#   QUANTIZER_TYPE=parallel K1=100 K2=500 sbatch slurm/causal_specunit/20_train_quantizer.sh
#
# Optional knobs:
#   STEPS=30000  BATCH=8192  DECAY=0.99  BETA=0.25
#   SOURCE_TARGETS_DIR=outputs/causal_specunit/targets_960h_c8
#   CHUNKS_CACHE=outputs/causal_specunit/targets_960h_c8/pca64_chunks.npy
#   OUTPUT_DIR_OVERRIDE=outputs/causal_specunit/vq/<custom-name>

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=train_vq
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/train_vq.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/train_vq.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
SOURCE_TARGETS_DIR="${SOURCE_TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"

QUANTIZER_TYPE="${QUANTIZER_TYPE:?must set QUANTIZER_TYPE to one of: vq, rvq, parallel}"
K1="${K1:?must set K1 (codebook size for vq, level-1 for rvq/parallel)}"
K2="${K2:-}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-8192}"
DECAY="${DECAY:-0.99}"
BETA="${BETA:-0.25}"
SEED="${SEED:-42}"

if [ -z "${OUTPUT_DIR_OVERRIDE:-}" ]; then
    if [ -n "${K2}" ]; then
        OUTPUT_DIR="outputs/causal_specunit/vq/${QUANTIZER_TYPE}_${K1}_${K2}"
    else
        OUTPUT_DIR="outputs/causal_specunit/vq/${QUANTIZER_TYPE}_${K1}"
    fi
else
    OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE}"
fi

CHUNKS_CACHE="${CHUNKS_CACHE:-${SOURCE_TARGETS_DIR}/pca64_chunks.npy}"

# Sanity checks
[ -d "${VIRTUAL_ENV}" ]                          || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -f "${SOURCE_TARGETS_DIR}/cluster_artifacts.joblib" ] || { echo "Missing ${SOURCE_TARGETS_DIR}/cluster_artifacts.joblib"; exit 1; }
[ -f "${SOURCE_TARGETS_DIR}/cmvn.pt" ]           || { echo "Missing ${SOURCE_TARGETS_DIR}/cmvn.pt"; exit 1; }
[ -d "${DATA_ROOT}/train-clean-100" ]            || { echo "Missing ${DATA_ROOT}/train-clean-100"; exit 1; }

case "${QUANTIZER_TYPE}" in
    vq) K2_ARG="" ;;
    rvq|parallel)
        [ -n "${K2}" ] || { echo "K2 is required for QUANTIZER_TYPE=${QUANTIZER_TYPE}"; exit 1; }
        K2_ARG="--K2 ${K2}"
        ;;
    *) echo "Unknown QUANTIZER_TYPE=${QUANTIZER_TYPE}"; exit 1 ;;
esac

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
mkdir -p logs "${OUTPUT_DIR}"

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] train_quantizer | type=${QUANTIZER_TYPE} K1=${K1} K2=${K2:-N/A}"
echo "  source=${SOURCE_TARGETS_DIR}"
echo "  cache=${CHUNKS_CACHE}"
echo "  output=${OUTPUT_DIR}"
echo "  steps=${STEPS} batch=${BATCH} decay=${DECAY} beta=${BETA} seed=${SEED}"
echo "===================================================="

python -m CausalSpecUnit.train_quantizer \
    --source-targets-dir "${SOURCE_TARGETS_DIR}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --chunks-cache "${CHUNKS_CACHE}" \
    --quantizer-type "${QUANTIZER_TYPE}" \
    --K1 "${K1}" ${K2_ARG} \
    --beta "${BETA}" --decay "${DECAY}" \
    --steps "${STEPS}" --batch-size "${BATCH}" \
    --seed "${SEED}"

echo "DONE: ${OUTPUT_DIR}/state.pt"
