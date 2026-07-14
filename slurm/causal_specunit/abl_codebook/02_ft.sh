#!/bin/bash
# Dual-codebook ablation: parameterized CTC fine-tune.
#
# Required env:
#   SSL_CHECKPOINT  full path to SSL checkpoint.pt
#   TRAIN_HOURS     10 or 100
#   FT_OUTPUT_DIR   where the fine-tune writes checkpoints + metrics
#
# Fine-tune recipe matches the fair 100h baseline (encoder_lr=3e-4, head_lr=1e-3,
# warmup 10, peak 50, decay 0.5) — chosen for direct comparability with the
# table baselines, not the more aggressive LP-FT recipe.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_abl_ft
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_abl_ft.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_abl_ft.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
TOKENIZER_PATH="dataset/bpe128.model"

: "${SSL_CHECKPOINT:?Required: SSL_CHECKPOINT path}"
: "${TRAIN_HOURS:?Required: TRAIN_HOURS (10 or 100)}"
: "${FT_OUTPUT_DIR:?Required: FT_OUTPUT_DIR}"

EPOCHS="${EPOCHS:-100}"
EVAL_SPLIT="${EVAL_SPLIT:-dev-other}"
SUBSET_SEED="${SUBSET_SEED:-42}"

# train-clean-100 is the source for both 10h and 100h.
# For 10h, --train-subset-hours samples ~10 audio hours with a fixed seed.
TRAIN_SPLITS="train-clean-100"
if [ "${TRAIN_HOURS}" = "10" ]; then
    SUBSET_ARG=(--train-subset-hours 10 --train-subset-seed "${SUBSET_SEED}")
elif [ "${TRAIN_HOURS}" = "100" ]; then
    SUBSET_ARG=()
else
    echo "Invalid TRAIN_HOURS=${TRAIN_HOURS} (must be 10 or 100)"; exit 1
fi

# Simple fine-tune recipe (matches the published fair 100h baseline):
#   - encoder LR 3e-4, head LR 1e-3 (single layer-decay = uniform encoder)
#   - Noam: warmup 10 ep, peak 50 ep, decay rate 0.5
#   - SpecAug with default zero-fill
#   - No InterCTC, no SSL anchor, no LP-FT
#
# IDENTICAL across all three codebook conditions — only the SSL pretraining
# objective varies. Keep this minimal so the ablation tests *the codebook
# objective*, not the optimizer.
ENCODER_LR="${ENCODER_LR:-3e-4}"
HEAD_LR="${HEAD_LR:-1e-3}"
BASE_LR="${BASE_LR:-1e-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
PEAK_EPOCHS="${PEAK_EPOCHS:-50}"
NOAM_DECAY_RATE="${NOAM_DECAY_RATE:-0.5}"

BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

NUM_PROCESSES=4
WORKERS=8

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
cd "${PROJECT_DIR}"
mkdir -p logs "${FT_OUTPUT_DIR}"

[ -f "${SSL_CHECKPOINT}" ] || { echo "Missing SSL checkpoint: ${SSL_CHECKPOINT}"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ] || { echo "Missing CMVN"; exit 1; }
[ -f "${TOKENIZER_PATH}" ] || { echo "Missing tokenizer"; exit 1; }

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((28000 + SLURM_JOB_ID % 20000))}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "Job ${SLURM_JOB_ID} | TRAIN_HOURS=${TRAIN_HOURS} | EPOCHS=${EPOCHS}"
echo "SSL_CHECKPOINT: ${SSL_CHECKPOINT}"
echo "FT_OUTPUT_DIR:  ${FT_OUTPUT_DIR}"
echo "Effective batch: $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
echo "Recipe: simple — encoder_lr=${ENCODER_LR}, head_lr=${HEAD_LR}, warmup=${WARMUP_EPOCHS}, peak=${PEAK_EPOCHS}, decay=${NOAM_DECAY_RATE}"

torchrun \
    --nproc_per_node="${NUM_PROCESSES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m CausalSpecUnit.train_ctc \
    --data-root "${DATA_ROOT}" \
    --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --train-splits ${TRAIN_SPLITS} \
    "${SUBSET_ARG[@]}" \
    --ssl-checkpoint "${SSL_CHECKPOINT}" \
    --output-dir "${FT_OUTPUT_DIR}" \
    --variant xs \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --eval-batch-size 128 \
    --eval-split "${EVAL_SPLIT}" \
    --eval-every 1 \
    --workers "${WORKERS}" \
    --dataloader-timeout 120 \
    --lr "${BASE_LR}" \
    --encoder-lr "${ENCODER_LR}" \
    --head-lr "${HEAD_LR}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --peak-epochs "${PEAK_EPOCHS}" \
    --noam-decay-rate "${NOAM_DECAY_RATE}" \
    --max-grad-norm 1.0 \
    --specaug \
    --specaug-time-mask-param 30 \
    --specaug-freq-mask-param 20 \
    --specaug-time-masks 2 \
    --specaug-freq-masks 2 \
    --specaug-disable-last-epochs 10 \
    --progress off \
    --log-every 0 \
    --save-every 10

echo "Done at $(date)"
echo "Best checkpoint: ${FT_OUTPUT_DIR}/checkpoint_best/checkpoint.pt"
