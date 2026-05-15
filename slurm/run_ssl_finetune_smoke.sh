#!/bin/bash
# 5-minute smoke test for the SSL fine-tune pipeline.
# Mirrors run_ssl_finetune.sh exactly, but caps at 2 epochs of 20 batches
# so it finishes quickly. Verifies:
#   - SSL checkpoint loads (--ssl-init)
#   - Encoder freeze logic works
#   - DDP + find_unused_parameters works on 2 GPUs
#   - metrics.csv gets written and flushed per epoch
#
# Submit and watch the .out file — if epoch 1 finishes cleanly and
# metrics.csv has at least 1 row, the full run will work too.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=sqformer_ssl_smoke
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/sqformer_ssl_smoke.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/sqformer_ssl_smoke.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
TOKENIZER_PATH="dataset/bpe128.model"
SSL_CHECKPOINT="outputs/causal_specunit/pretrain_ssl_50k_c2_v2/checkpoint_step050000/checkpoint.pt"
OUTPUT_DIR="outputs/squeezeformer_xs_ssl_smoke"

if [ ! -d "${VIRTUAL_ENV}" ]; then
    echo "Missing venv: ${VIRTUAL_ENV}"
    exit 1
fi

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"
# Wipe any prior smoke-test artifacts so we get a fresh run.
rm -f "${OUTPUT_DIR}/metrics.csv"

if [ ! -f "${TOKENIZER_PATH}" ]; then
    echo "Missing tokenizer: ${TOKENIZER_PATH}"
    exit 1
fi

if [ ! -f "${SSL_CHECKPOINT}" ]; then
    echo "Missing SSL checkpoint: ${SSL_CHECKPOINT}"
    exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
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

echo "Job ${SLURM_JOB_ID} SMOKE TEST starting at $(date)"
echo "SSL checkpoint: ${SSL_CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"

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
    --epochs 2 \
    --max-train-batches 20 \
    --variant xs \
    --eval-split dev-other \
    --eval-every 1 \
    --no-compile \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --batch-size 64 \
    --grad-accum-steps 1 \
    --lr 5e-4 \
    --warmup-epochs 1 \
    --peak-epochs 1 \
    --noam-decay-rate 1.0 \
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 200.0 \
    --freeze-encoder-epochs 1 \
    --eval-batch-size 64 \
    --workers "${WORKERS}" \
    --log-every 5 \
    --train-metrics-every 0 \
    --progress on \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --output-dir "${OUTPUT_DIR}" \
    --ssl-init "${SSL_CHECKPOINT}" \
    --run-name squeezeformer_xs_ssl_smoke

echo "Job ${SLURM_JOB_ID} SMOKE TEST finished at $(date)"
echo ""
echo "=== metrics.csv ==="
cat "${OUTPUT_DIR}/metrics.csv" || echo "(metrics.csv missing)"