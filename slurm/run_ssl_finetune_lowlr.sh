#!/bin/bash
# SSL fine-tune salvage run for the existing 50k c2 checkpoint.
# This is not the recommended main paper run; use it to test whether the
# old checkpoint becomes useful once the fine-tune frontend matches SSL CMVN.
#
# Rationale: the default fine-tune recipe (lr=1e-3, warmup=20 epochs) was
# tuned for from-scratch training. SSL features have inertia and can be
# overwritten by aggressive LR. We lower the peak LR and shorten warmup
# so the CTC head reaches its operating LR quickly without destroying
# the pretrained encoder.
#
# Apples-to-apples claim for this salvage row: compare only against a scratch
# run using the same CMVN frontend and SqueezeFormer/train.py code path.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=sqformer_xs_ssl_lowlr
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/sqformer_xs_ssl_lowlr.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/sqformer_xs_ssl_lowlr.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
TOKENIZER_PATH="dataset/bpe128.model"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c2"
SSL_CHECKPOINT="outputs/causal_specunit/pretrain_ssl_50k_c2_v2/checkpoint_step050000/checkpoint.pt"
OUTPUT_DIR="outputs/squeezeformer_xs_150ep_ssl_50k_c2_cmvn_salvage"

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
    exit 1
fi

if [ ! -f "${TARGETS_DIR}/cmvn.pt" ]; then
    echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"
    exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((16000 + SLURM_JOB_ID % 20000))}"
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
echo "Project: ${PROJECT_DIR}"
echo "Python: $(which python)"
echo "Torchrun: $(which torchrun)"
echo "SSL checkpoint: ${SSL_CHECKPOINT}"
echo "CMVN: ${TARGETS_DIR}/cmvn.pt"
echo "GPUs on node: ${NUM_PROCESSES}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Recipe: salvage old c2 SSL, CMVN-matched frontend, no freeze, encoder/head LR split"

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
    --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
    --epochs 150 \
    --variant xs \
    --eval-split dev-other \
    --eval-every 1 \
    --no-compile \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --batch-size 128 \
    --grad-accum-steps 2 \
    --lr 1e-3 \
    --encoder-lr 2e-4 \
    --head-lr 1e-3 \
    --warmup-epochs 5 \
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
    --run-name squeezeformer_xs_150ep_ssl_50k_c2_cmvn_salvage

echo "Job ${SLURM_JOB_ID} finished at $(date)"
