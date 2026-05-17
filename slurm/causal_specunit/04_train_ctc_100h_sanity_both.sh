#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ctc100h_both
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc100h_both.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc100h_both.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-dataset/bpe128.model}"
SSL_CHECKPOINT="${SSL_CHECKPOINT:-outputs/causal_specunit/pretrain_ssl_100k_c8/checkpoint_step100000/checkpoint.pt}"

# Default to train-clean-100 so the 100h condition is the canonical
# LibriSpeech low-resource split. Override TRAIN_SPLITS if needed.
TRAIN_SPLITS="${TRAIN_SPLITS:-train-clean-100}"
TRAIN_HOURS="${TRAIN_HOURS:-100}"
SUBSET_SEED="${SUBSET_SEED:-42}"
EVAL_SPLIT="${EVAL_SPLIT:-dev-other}"
EPOCHS="${EPOCHS:-150}"

BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
WORKERS="${WORKERS:-8}"
DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-120}"

SSL_OUTPUT_DIR="${SSL_OUTPUT_DIR:-outputs/causal_specunit/ctc_ssl_100h_150ep_c8}"
SCRATCH_OUTPUT_DIR="${SCRATCH_OUTPUT_DIR:-outputs/causal_specunit/ctc_scratch_100h_150ep_c8}"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "${PROJECT_DIR}"
mkdir -p logs "${SSL_OUTPUT_DIR}" "${SCRATCH_OUTPUT_DIR}"

if [ ! -d "${VIRTUAL_ENV}" ]; then
    echo "Missing venv: ${VIRTUAL_ENV}"
    exit 1
fi
if [ ! -d "${DATA_ROOT}" ]; then
    echo "Missing data root: ${DATA_ROOT}"
    exit 1
fi
if [ ! -f "${TARGETS_DIR}/cmvn.pt" ]; then
    echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"
    exit 1
fi
if [ ! -f "${TOKENIZER_PATH}" ]; then
    echo "Missing tokenizer: ${TOKENIZER_PATH}"
    exit 1
fi
if [ ! -f "${SSL_CHECKPOINT}" ]; then
    echo "Missing SSL checkpoint: ${SSL_CHECKPOINT}"
    exit 1
fi

read -r -a TRAIN_SPLIT_ARGS <<< "${TRAIN_SPLITS}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((16000 + SLURM_JOB_ID % 20000))}"

NUM_PROCESSES=2

echo "Job ${SLURM_JOB_ID} 100h CTC sanity starting at $(date)"
echo "Python: $(which python)"
echo "Torchrun: $(which torchrun)"
echo "Data root: ${DATA_ROOT}"
echo "Train splits: ${TRAIN_SPLITS}"
echo "Train hours: ${TRAIN_HOURS}"
echo "Eval split: ${EVAL_SPLIT}"
echo "SSL output: ${SSL_OUTPUT_DIR}"
echo "Scratch output: ${SCRATCH_OUTPUT_DIR}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY

COMMON_ARGS=(
    -m CausalSpecUnit.train_ctc
    --data-root "${DATA_ROOT}"
    --cmvn-path "${TARGETS_DIR}/cmvn.pt"
    --tokenizer-path "${TOKENIZER_PATH}"
    --variant xs
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --eval-split "${EVAL_SPLIT}"
    --eval-every 1
    --workers "${WORKERS}"
    --dataloader-timeout "${DATALOADER_TIMEOUT}"
    --train-subset-hours "${TRAIN_HOURS}"
    --train-subset-seed "${SUBSET_SEED}"
    --train-splits "${TRAIN_SPLIT_ARGS[@]}"
    --progress off
    --log-every 0
    --save-every 10
)

echo "Starting SSL 100h sanity run at $(date)"
torchrun \
    --nproc_per_node="${NUM_PROCESSES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${COMMON_ARGS[@]}" \
    --ssl-checkpoint "${SSL_CHECKPOINT}" \
    --output-dir "${SSL_OUTPUT_DIR}" \
    --lr 1e-3 \
    --encoder-lr 3e-4 \
    --head-lr 1e-3

echo "Starting scratch 100h sanity run at $(date)"
export MASTER_PORT="$((MASTER_PORT + 1))"
torchrun \
    --nproc_per_node="${NUM_PROCESSES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${COMMON_ARGS[@]}" \
    --output-dir "${SCRATCH_OUTPUT_DIR}" \
    --lr 1e-3

echo "Job ${SLURM_JOB_ID} 100h CTC sanity finished at $(date)"
echo "Metrics:"
echo "  ${SSL_OUTPUT_DIR}/ctc_metrics.jsonl"
echo "  ${SCRATCH_OUTPUT_DIR}/ctc_metrics.jsonl"
