#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=sqformer_xs
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/sqformer_xs.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/sqformer_xs.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
TOKENIZER_PATH="dataset/bpe128.model"
OUTPUT_DIR="outputs/squeezeformer_xs_150ep_scratch"

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

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((12000 + SLURM_JOB_ID % 20000))}"
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

# Match the Slurm request above. Keeping this explicit avoids accidentally
# launching 8 processes when Slurm reports no GPU count variable.
NUM_PROCESSES=2
WORKERS=8
DATALOADER_TIMEOUT=120

echo "Job ${SLURM_JOB_ID} starting at $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Python: $(which python)"
echo "Torchrun: $(which torchrun)"
echo "GPUs on node: ${NUM_PROCESSES}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

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
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 50.0 \
    --eval-batch-size 128 \
    --workers "${WORKERS}" \
    --log-every 0 \
    --train-metrics-every 0 \
    --progress on \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name squeezeformer_xs_150ep_scratch

echo "Job ${SLURM_JOB_ID} finished at $(date)"
