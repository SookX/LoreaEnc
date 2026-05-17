#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ctc100h_aug_ssl
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc100h_aug_ssl.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc100h_aug_ssl.%j.err

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

TRAIN_SPLITS="${TRAIN_SPLITS:-train-clean-100}"
TRAIN_HOURS="${TRAIN_HOURS:-100}"
SUBSET_SEED="${SUBSET_SEED:-42}"
EVAL_SPLIT="${EVAL_SPLIT:-dev-other}"
EPOCHS="${EPOCHS:-100}"

BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
WORKERS="${WORKERS:-8}"
DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-120}"

SSL_ENCODER_LR="${SSL_ENCODER_LR:-5e-4}"
SSL_HEAD_LR="${SSL_HEAD_LR:-1e-3}"
SSL_BASE_LR="${SSL_BASE_LR:-1e-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
PEAK_EPOCHS="${PEAK_EPOCHS:-50}"
NOAM_DECAY_RATE="${NOAM_DECAY_RATE:-0.5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

SPECAUG_TIME_MASK_PARAM="${SPECAUG_TIME_MASK_PARAM:-40}"
SPECAUG_FREQ_MASK_PARAM="${SPECAUG_FREQ_MASK_PARAM:-30}"
SPECAUG_TIME_MASKS="${SPECAUG_TIME_MASKS:-2}"
SPECAUG_FREQ_MASKS="${SPECAUG_FREQ_MASKS:-2}"
SPECAUG_DISABLE_LAST_EPOCHS="${SPECAUG_DISABLE_LAST_EPOCHS:-15}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/ctc_ssl_100h_specaug_elr5e4_w10_p50_100ep_c8}"

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
mkdir -p logs "${OUTPUT_DIR}"

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
export MASTER_PORT="${MASTER_PORT:-$((19000 + SLURM_JOB_ID % 20000))}"

NUM_PROCESSES=2

echo "Job ${SLURM_JOB_ID} 100h SSL SpecAug run starting at $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "SSL checkpoint: ${SSL_CHECKPOINT}"
echo "Train splits: ${TRAIN_SPLITS} | hours=${TRAIN_HOURS} | eval=${EVAL_SPLIT}"
echo "SpecAug: time=${SPECAUG_TIME_MASK_PARAM}x${SPECAUG_TIME_MASKS} freq=${SPECAUG_FREQ_MASK_PARAM}x${SPECAUG_FREQ_MASKS} disable_last=${SPECAUG_DISABLE_LAST_EPOCHS}"
echo "Schedule: encoder_lr=${SSL_ENCODER_LR} head_lr=${SSL_HEAD_LR} warmup=${WARMUP_EPOCHS} hold=${PEAK_EPOCHS} decay=${NOAM_DECAY_RATE}"

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
    -m CausalSpecUnit.train_ctc \
    --data-root "${DATA_ROOT}" \
    --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
    --ssl-checkpoint "${SSL_CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --variant xs \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --eval-split "${EVAL_SPLIT}" \
    --eval-every 1 \
    --workers "${WORKERS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --train-subset-hours "${TRAIN_HOURS}" \
    --train-subset-seed "${SUBSET_SEED}" \
    --train-splits "${TRAIN_SPLIT_ARGS[@]}" \
    --lr "${SSL_BASE_LR}" \
    --encoder-lr "${SSL_ENCODER_LR}" \
    --head-lr "${SSL_HEAD_LR}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --peak-epochs "${PEAK_EPOCHS}" \
    --noam-decay-rate "${NOAM_DECAY_RATE}" \
    --max-grad-norm "${MAX_GRAD_NORM}" \
    --specaug \
    --specaug-time-mask-param "${SPECAUG_TIME_MASK_PARAM}" \
    --specaug-freq-mask-param "${SPECAUG_FREQ_MASK_PARAM}" \
    --specaug-time-masks "${SPECAUG_TIME_MASKS}" \
    --specaug-freq-masks "${SPECAUG_FREQ_MASKS}" \
    --specaug-disable-last-epochs "${SPECAUG_DISABLE_LAST_EPOCHS}" \
    --log-every 0 \
    --save-every 10 \
    --progress off

echo "Job ${SLURM_JOB_ID} 100h SSL SpecAug run finished at $(date)"
echo "Metrics: ${OUTPUT_DIR}/ctc_metrics.jsonl"
