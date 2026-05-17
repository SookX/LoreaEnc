#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_iter2_targets_c8
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_iter2_targets_c8.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_iter2_targets_c8.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
SOURCE_TARGETS_DIR="${SOURCE_TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
SOURCE_SSL_CHECKPOINT="${SOURCE_SSL_CHECKPOINT:-outputs/causal_specunit/pretrain_ssl_100k_c8/checkpoint_step100000/checkpoint.pt}"
ITER2_TARGETS_DIR="${ITER2_TARGETS_DIR:-outputs/causal_specunit/targets_iter2_ssl100k_c8}"

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

if [ ! -d "${VIRTUAL_ENV}" ]; then
    echo "Missing venv: ${VIRTUAL_ENV}"
    exit 1
fi
if [ ! -d "${DATA_ROOT}" ]; then
    echo "Missing data root: ${DATA_ROOT}"
    exit 1
fi
if [ ! -f "${SOURCE_TARGETS_DIR}/cmvn.pt" ]; then
    echo "Missing source CMVN: ${SOURCE_TARGETS_DIR}/cmvn.pt"
    exit 1
fi
if [ ! -f "${SOURCE_SSL_CHECKPOINT}" ] && [ ! -f "${SOURCE_SSL_CHECKPOINT}/checkpoint.pt" ]; then
    echo "Missing source SSL checkpoint: ${SOURCE_SSL_CHECKPOINT}"
    exit 1
fi

echo "Job ${SLURM_JOB_ID} iter-2 target generation starting at $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Python: $(which python)"
echo "Data root: ${DATA_ROOT}"
echo "Source targets: ${SOURCE_TARGETS_DIR}"
echo "Source SSL checkpoint: ${SOURCE_SSL_CHECKPOINT}"
echo "Iter-2 targets: ${ITER2_TARGETS_DIR}"
echo "Fit frames: ${MAX_FIT_FRAMES}"

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
import json
import os

targets_dir = os.environ.get("ITER2_TARGETS_DIR", "outputs/causal_specunit/targets_iter2_ssl100k_c8")
with open(os.path.join(targets_dir, "metadata.json"), encoding="utf-8") as f:
    metadata = json.load(f)
print("Iter-2 target metadata:", {
    "target_features": metadata.get("target_features"),
    "num_target_utterances": metadata.get("num_target_utterances"),
    "num_encoder_frames": metadata.get("num_encoder_frames"),
    "num_fit_frames": metadata.get("num_fit_frames"),
    "elapsed_hours": metadata.get("elapsed_hours"),
})
PY

echo "Job ${SLURM_JOB_ID} iter-2 target generation finished at $(date)"
