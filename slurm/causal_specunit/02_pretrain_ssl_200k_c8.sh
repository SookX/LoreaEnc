#!/bin/bash
# Old SSL recipe with the only meaningful improvement: train 2x longer.
#
# Identical to 02_pretrain_ssl_100k_c8.sh except:
#   --max-steps 100000 → 200000     (2x training compute)
#   --peak-epochs 20 → 40           (more time at peak LR before decay,
#                                    proportional to 2x total budget)
#   Output dir → pretrain_ssl_200k_c8/  (separate from 100k checkpoint)
#
# Everything else — mask_prob 0.30, mask_length 10, chunk_size/stride 8/4,
# lr 1e-3, warmup 20 ep, no SpecAug during SSL, no LayerDrop, no aux heads,
# no teacher — is identical to the proven 100k recipe. The only thing that
# changes is "the encoder gets twice as long to learn from the same targets".

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ssl200k_c8
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ssl200k_c8.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ssl200k_c8.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/pretrain_ssl_200k_c8}"

MAX_STEPS="${MAX_STEPS:-200000}"
PEAK_EPOCHS="${PEAK_EPOCHS:-40}"

if [ ! -d "${VIRTUAL_ENV}" ]; then
    echo "Missing venv: ${VIRTUAL_ENV}"
    exit 1
fi

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${DATA_ROOT}" ]; then
    echo "Missing data root: ${DATA_ROOT}"; exit 1
fi
if [ ! -f "${TARGETS_DIR}/targets.pt" ]; then
    echo "Missing targets: ${TARGETS_DIR}/targets.pt"; exit 1
fi
if [ ! -f "${TARGETS_DIR}/cmvn.pt" ]; then
    echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"; exit 1
fi
if [ ! -f "${TARGETS_DIR}/metadata.json" ]; then
    echo "Missing metadata: ${TARGETS_DIR}/metadata.json"; exit 1
fi
if [ ! -f "${TARGETS_DIR}/target_index.json" ]; then
    echo "Missing sharded target index: ${TARGETS_DIR}/target_index.json"
    python -m CausalSpecUnit.shard_targets --targets-dir "${TARGETS_DIR}" --num-shards 128
fi

export TARGETS_DIR
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((13500 + SLURM_JOB_ID % 20000))}"
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

NUM_PROCESSES=2
WORKERS=12
DATALOADER_TIMEOUT=300

echo "Job ${SLURM_JOB_ID} 200k SSL pretraining starting at $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Targets: ${TARGETS_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Max steps: ${MAX_STEPS}"
echo "Schedule: warmup=20 peak=${PEAK_EPOCHS} decay_rate=1.0"

python - <<'PY'
import json, os, torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
with open(os.path.join(os.environ["TARGETS_DIR"], "metadata.json"), encoding="utf-8") as f:
    metadata = json.load(f)
print("Target metadata:", {k: metadata.get(k) for k in
      ["chunk_size","chunk_stride","pca_dim","k_coarse","k_fine","target_features","num_target_utterances"]})
PY

# 8h wall-time cap: resume from the latest checkpoint and queue a successor so
# the run completes across a chain of 8h jobs. Exits here if MAX_STEPS is
# already reached. Sets RESUME_CKPT, consumed by the block just below.
SELF_SCRIPT="slurm/causal_specunit/02_pretrain_ssl_200k_c8.sh"
source slurm/causal_specunit/_autochain.sh

RESUME_CKPT="${RESUME_CKPT:-}"
if [ -n "${RESUME_CKPT}" ]; then
    if [ ! -f "${RESUME_CKPT}/checkpoint.pt" ]; then
        echo "RESUME_CKPT set but checkpoint.pt not found: ${RESUME_CKPT}"
        exit 1
    fi
    echo "Resuming from checkpoint: ${RESUME_CKPT}"
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
    --batch-size 128 \
    --grad-accum-steps 1 \
    --mask-prob 0.30 \
    --mask-length 10 \
    --chunk-size 8 \
    --chunk-stride 4 \
    --lr 1e-3 \
    --warmup-epochs 20 \
    --peak-epochs "${PEAK_EPOCHS}" \
    --noam-decay-rate 1.0 \
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 200.0 \
    --workers "${WORKERS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --prefetch-factor 4 \
    --log-every 10 \
    --save-every 1 \
    --trace-startup \
    --progress on \
    $( [ -n "${RESUME_CKPT}" ] && echo "--resume ${RESUME_CKPT}" || true )

echo "Job ${SLURM_JOB_ID} 200k SSL pretraining finished at $(date)"
echo "Output checkpoint: ${OUTPUT_DIR}/checkpoint_step$(printf '%06d' ${MAX_STEPS})/checkpoint.pt"
