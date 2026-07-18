#!/bin/bash
# Generate SSL targets (global CMVN + PCA-64 + dual k-means K=100/500) for one
# MLS language, using the SAME recipe as the English targets_960h_c8 run.
# CPU-bound (mel extraction + PCA + MiniBatchKMeans); the GPU is requested only
# because the QOS mandates one (QOSMinGRES). OMP threads speed up mel FFTs.
#
# CMVN is resumable: if cmvn.pt already exists in the output dir, a re-submit
# skips straight to chunk collection / assignment.
#
# Submit:
#   MLS_LANG=polish     sbatch slurm/causal_specunit/61_mls_targets.sh
#   MLS_LANG=portuguese sbatch slurm/causal_specunit/61_mls_targets.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=mls_targets
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/mls_targets.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/mls_targets.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"

MLS_LANG="${MLS_LANG:-polish}"
MLS_LANG_ROOT="${MLS_LANG_ROOT:-dataset/mls/mls_${MLS_LANG}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/mls_targets_${MLS_LANG}_c8}"
TARGET_SHARDS="${TARGET_SHARDS:-128}"

[ -d "${VIRTUAL_ENV}" ]                     || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONUNBUFFERED=1
# Let torch use the allocated cores for the mel-spectrogram FFTs (single process).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
mkdir -p logs "${OUTPUT_DIR}"

[ -d "${MLS_LANG_ROOT}/train/audio" ]       || { echo "Missing MLS audio: ${MLS_LANG_ROOT}/train/audio"; exit 1; }
[ -f "${MLS_LANG_ROOT}/train/transcripts.txt" ] || { echo "Missing transcripts: ${MLS_LANG_ROOT}/train/transcripts.txt"; exit 1; }

echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] MLS targets: ${MLS_LANG}"
echo "  mls_lang_root=${MLS_LANG_ROOT}"
echo "  output=${OUTPUT_DIR}  shards=${TARGET_SHARDS}  OMP=${OMP_NUM_THREADS}"
echo "===================================================="

python -m CausalSpecUnit.generate_targets \
    --mls-lang-root "${MLS_LANG_ROOT}" \
    --splits train \
    --output-dir "${OUTPUT_DIR}" \
    --chunk-size 8 \
    --chunk-stride 4 \
    --pca-dim 64 \
    --k-coarse 100 \
    --k-fine 500 \
    --max-fit-chunks 1000000 \
    --target-shards "${TARGET_SHARDS}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE. Targets in ${OUTPUT_DIR}"
ls -1 "${OUTPUT_DIR}"
echo ""
echo "Next: SSL pretrain ->"
echo "  MLS_LANG=${MLS_LANG} sbatch slurm/causal_specunit/62_mls_pretrain.sh"
