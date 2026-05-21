#!/bin/bash
# Iter-2 target generation from the v2 iter-1 checkpoint.
#
# Differs from 01d_generate_iter2_targets_c8.sh only in defaults:
#   SOURCE_SSL_CHECKPOINT -> the v2 iter-1 (150k steps, new SSL recipe)
#   ITER2_TARGETS_DIR     -> targets_iter2_v2_c8/  (separate dir; the old
#                            targets_iter2_ssl100k_c8/ stays untouched)

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_iter2_v2_targets_c8
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_iter2_v2_targets_c8.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_iter2_v2_targets_c8.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
SOURCE_TARGETS_DIR="${SOURCE_TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
SOURCE_SSL_CHECKPOINT="${SOURCE_SSL_CHECKPOINT:-outputs/causal_specunit/pretrain_ssl_v2_150k_c8/checkpoint_step150000/checkpoint.pt}"
ITER2_TARGETS_DIR="${ITER2_TARGETS_DIR:-outputs/causal_specunit/targets_iter2_v2_c8}"

MAX_FIT_FRAMES="${MAX_FIT_FRAMES:-1000000}"
FIT_FRAMES_PER_BATCH="${FIT_FRAMES_PER_BATCH:-8192}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-8}"
DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-180}"
TARGET_SHARDS="${TARGET_SHARDS:-128}"
SEED="${SEED:-42}"

export VIRTUAL_ENV
export ITER2_TARGETS_DIR
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

cd "${PROJECT_DIR}"
mkdir -p logs "${ITER2_TARGETS_DIR}"

[ -d "${VIRTUAL_ENV}" ] || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}" ]   || { echo "Missing data root: ${DATA_ROOT}"; exit 1; }
[ -f "${SOURCE_TARGETS_DIR}/cmvn.pt" ] || { echo "Missing source CMVN: ${SOURCE_TARGETS_DIR}/cmvn.pt"; exit 1; }
if [ ! -f "${SOURCE_SSL_CHECKPOINT}" ] && [ ! -f "${SOURCE_SSL_CHECKPOINT}/checkpoint.pt" ]; then
    echo "Missing source SSL checkpoint: ${SOURCE_SSL_CHECKPOINT}"; exit 1
fi

# Idempotency: if iter-2 targets are already finished, skip.
if [ -f "${ITER2_TARGETS_DIR}/targets.pt" ] && [ -f "${ITER2_TARGETS_DIR}/metadata.json" ]; then
    echo "Iter-2 targets already exist at ${ITER2_TARGETS_DIR}, skipping."
    exit 0
fi

echo "Job ${SLURM_JOB_ID} iter-2 v2 target generation starting at $(date)"
echo "Source iter-1 (v2) checkpoint: ${SOURCE_SSL_CHECKPOINT}"
echo "Source CMVN:                   ${SOURCE_TARGETS_DIR}/cmvn.pt"
echo "Iter-2 targets dir:            ${ITER2_TARGETS_DIR}"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY

python -m CausalSpecUnit.generate_iter2_targets \
    --data-root "${DATA_ROOT}" \
    --splits train-clean-100 train-clean-360 train-other-500 \
    --cmvn-path "${SOURCE_TARGETS_DIR}/cmvn.pt" \
    --ssl-checkpoint "${SOURCE_SSL_CHECKPOINT}" \
    --output-dir "${ITER2_TARGETS_DIR}" \
    --variant xs \
    --chunk-size 8 \
    --chunk-stride 4 \
    --pca-dim 64 \
    --k-coarse 100 \
    --k-fine 500 \
    --max-fit-frames "${MAX_FIT_FRAMES}" \
    --fit-frames-per-batch "${FIT_FRAMES_PER_BATCH}" \
    --batch-size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --target-shards "${TARGET_SHARDS}" \
    --seed "${SEED}"

python - <<'PY'
import json, os
targets_dir = os.environ["ITER2_TARGETS_DIR"]
with open(os.path.join(targets_dir, "metadata.json"), encoding="utf-8") as f:
    metadata = json.load(f)
print("Iter-2 (v2) target metadata:", {
    "target_features": metadata.get("target_features"),
    "num_target_utterances": metadata.get("num_target_utterances"),
    "num_encoder_frames": metadata.get("num_encoder_frames"),
    "num_fit_frames": metadata.get("num_fit_frames"),
    "elapsed_hours": metadata.get("elapsed_hours"),
})
PY

echo "Job ${SLURM_JOB_ID} iter-2 v2 target generation finished at $(date)"
