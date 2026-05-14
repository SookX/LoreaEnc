#!/bin/bash
# SSL fine-tune from the older chunk_size=4 / chunk_stride=4 SSL checkpoint.
# This checkpoint pre-dates the chunk_size=2 mask-leakage bug in
# corrupt_mel_from_target_mask. With c=s=4, the buggy line of code
# happens to produce the correct mask, so this checkpoint is bug-free.
#
# Apples-to-apples vs run_baseline.sh: identical model/data/optimizer,
# only difference is encoder initialization (--ssl-init).

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=sqformer_xs_ssl_c4
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/sqformer_xs_ssl_c4.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/sqformer_xs_ssl_c4.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
TOKENIZER_PATH="dataset/bpe128.model"
# Bug-free chunk_size=4/stride=4 SSL checkpoint from earlier pretrain.
# Update the step number to match the actual final checkpoint in the dir.
SSL_CHECKPOINT="outputs/causal_specunit/pretrain_ssl_150k/checkpoint_step100000/checkpoint.pt"
OUTPUT_DIR="outputs/squeezeformer_xs_150ep_ssl_c4"

if [ ! -d "${VIRTUAL_ENV}" ]; then
    echo "Missing venv: ${VIRTUAL_ENV}"
    exit 1
fi

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -f "${TOKENIZER_PATH}" ]; then
    echo "Missing tokenizer: ${TOKENIZER_PATH}"
    exit 1
fi

if [ ! -f "${SSL_CHECKPOINT}" ]; then
    echo "Missing SSL checkpoint: ${SSL_CHECKPOINT}"
    echo "Update SSL_CHECKPOINT to the actual final step in pretrain_ssl_150k/."
    echo "Available checkpoints:"
    ls -d outputs/causal_specunit/pretrain_ssl_150k/checkpoint_step* 2>/dev/null || echo "  (none found)"
    exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((18000 + SLURM_JOB_ID % 20000))}"
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

NUM_PROCESSES=2
WORKERS=8
DATALOADER_TIMEOUT=120

echo "Job ${SLURM_JOB_ID} starting at $(date)"
echo "SSL checkpoint: ${SSL_CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
echo "Recipe: identical to run_baseline.sh, only --ssl-init differs"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY

torchrun \
    --nproc_per_node="${NUM_PROCESSES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    SqueezeFormer/train.py \
    --data-root dataset/datasets/librispeech/LibriSpeech \
    --epochs 150 \
    --variant xs \
    --eval-split dev-other \
    --eval-every 1 \
    --no-compile \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --batch-size 128 \
    --grad-accum-steps 2 \
    --lr 1e-3 \
    --warmup-epochs 20 \
    --peak-epochs 20 \
    --noam-decay-rate 1.0 \
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 200.0 \
    --eval-batch-size 128 \
    --workers "${WORKERS}" \
    --log-every 0 \
    --train-metrics-every 0 \
    --progress on \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --output-dir "${OUTPUT_DIR}" \
    --ssl-init "${SSL_CHECKPOINT}" \
    --run-name squeezeformer_xs_150ep_ssl_c4

echo "Job ${SLURM_JOB_ID} finished at $(date)"