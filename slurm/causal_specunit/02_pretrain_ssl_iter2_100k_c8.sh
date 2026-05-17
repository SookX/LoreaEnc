#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ssl_iter2_100k_c8
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ssl_iter2_100k_c8.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ssl_iter2_100k_c8.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_iter2_ssl100k_c8}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/pretrain_ssl_iter2_100k_c8}"
SOURCE_SSL_CHECKPOINT="${SOURCE_SSL_CHECKPOINT:-outputs/causal_specunit/pretrain_ssl_100k_c8/checkpoint_step100000/checkpoint.pt}"

# By default this warm-starts only the encoder and mask token from iter-1, while
# keeping the iter-2 SSL prediction heads fresh. To force a strict from-scratch
# iter-2 model, submit with: INIT_ENCODER_CHECKPOINT="" sbatch ...
INIT_ENCODER_CHECKPOINT="${INIT_ENCODER_CHECKPOINT-__DEFAULT__}"
if [ "${INIT_ENCODER_CHECKPOINT}" = "__DEFAULT__" ]; then
    INIT_ENCODER_CHECKPOINT="${SOURCE_SSL_CHECKPOINT}"
fi

NUM_PROCESSES="${NUM_PROCESSES:-2}"
WORKERS="${WORKERS:-12}"
DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-300}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_STEPS="${MAX_STEPS:-100000}"
LR="${LR:-1e-3}"
MASK_PROB="${MASK_PROB:-0.30}"
MASK_LENGTH="${MASK_LENGTH:-10}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
PEAK_EPOCHS="${PEAK_EPOCHS:-20}"
NOAM_DECAY_RATE="${NOAM_DECAY_RATE:-1.0}"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${VIRTUAL_ENV}" ]; then
    echo "Missing venv: ${VIRTUAL_ENV}"
    exit 1
fi
if [ ! -d "${DATA_ROOT}" ]; then
    echo "Missing data root: ${DATA_ROOT}"
    exit 1
fi
if [ ! -f "${TARGETS_DIR}/targets.pt" ]; then
    echo "Missing iter-2 targets: ${TARGETS_DIR}/targets.pt"
    echo "Run slurm/causal_specunit/01d_generate_iter2_targets_c8.sh first."
    exit 1
fi
if [ ! -f "${TARGETS_DIR}/cmvn.pt" ]; then
    echo "Missing iter-2 CMVN: ${TARGETS_DIR}/cmvn.pt"
    exit 1
fi
if [ ! -f "${TARGETS_DIR}/metadata.json" ]; then
    echo "Missing iter-2 metadata: ${TARGETS_DIR}/metadata.json"
    exit 1
fi
if [ -n "${INIT_ENCODER_CHECKPOINT}" ] \
    && [ ! -f "${INIT_ENCODER_CHECKPOINT}" ] \
    && [ ! -f "${INIT_ENCODER_CHECKPOINT}/checkpoint.pt" ]; then
    echo "Missing init encoder checkpoint: ${INIT_ENCODER_CHECKPOINT}"
    exit 1
fi

if [ ! -f "${TARGETS_DIR}/target_index.json" ]; then
    echo "Missing sharded target index: ${TARGETS_DIR}/target_index.json"
    echo "Creating sharded targets from ${TARGETS_DIR}/targets.pt"
    python -m CausalSpecUnit.shard_targets \
        --targets-dir "${TARGETS_DIR}" \
        --num-shards 128
fi

export TARGETS_DIR
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((15000 + SLURM_JOB_ID % 20000))}"
export PYTHONFAULTHANDLER=1
export PYTHONFAULTHANDLER_TIMEOUT=300
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "Job ${SLURM_JOB_ID} iter-2 SSL pretraining starting at $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Python: $(which python)"
echo "Torchrun: $(which torchrun)"
echo "Data root: ${DATA_ROOT}"
echo "Targets: ${TARGETS_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Init encoder checkpoint: ${INIT_ENCODER_CHECKPOINT:-none}"
echo "GPUs on node: ${NUM_PROCESSES}"
echo "Workers per rank: ${WORKERS}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

python - <<'PY'
import json
import os
import torch

targets_dir = os.environ["TARGETS_DIR"]
metadata_path = os.path.join(targets_dir, "metadata.json")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
with open(metadata_path, encoding="utf-8") as f:
    metadata = json.load(f)
print("Target metadata:", {
    "target_features": metadata.get("target_features"),
    "chunk_size": metadata.get("chunk_size"),
    "chunk_stride": metadata.get("chunk_stride"),
    "pca_dim": metadata.get("pca_dim"),
    "k_coarse": metadata.get("k_coarse"),
    "k_fine": metadata.get("k_fine"),
    "num_target_utterances": metadata.get("num_target_utterances"),
    "num_encoder_frames": metadata.get("num_encoder_frames"),
})
PY

RESUME_CKPT="${RESUME_CKPT:-}"
if [ -n "${RESUME_CKPT}" ]; then
    if [ ! -f "${RESUME_CKPT}/checkpoint.pt" ]; then
        echo "RESUME_CKPT set but checkpoint.pt not found: ${RESUME_CKPT}"
        exit 1
    fi
    echo "Resuming from checkpoint: ${RESUME_CKPT}"
fi

EXTRA_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
    EXTRA_ARGS+=(--resume "${RESUME_CKPT}")
elif [ -n "${INIT_ENCODER_CHECKPOINT}" ]; then
    EXTRA_ARGS+=(--init-encoder-checkpoint "${INIT_ENCODER_CHECKPOINT}")
fi

torchrun \
    --nproc_per_node="${NUM_PROCESSES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m CausalSpecUnit.pretrain_ssl \
    --data-root "${DATA_ROOT}" \
    --targets-dir "${TARGETS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --variant xs \
    --epochs 1000 \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --mask-prob "${MASK_PROB}" \
    --mask-length "${MASK_LENGTH}" \
    --chunk-size 8 \
    --chunk-stride 4 \
    --lr "${LR}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --peak-epochs "${PEAK_EPOCHS}" \
    --noam-decay-rate "${NOAM_DECAY_RATE}" \
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 200.0 \
    --workers "${WORKERS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --prefetch-factor 4 \
    --log-every 10 \
    --save-every 10 \
    --trace-startup \
    --progress on \
    "${EXTRA_ARGS[@]}"

echo "Job ${SLURM_JOB_ID} iter-2 SSL pretraining finished at $(date)"
