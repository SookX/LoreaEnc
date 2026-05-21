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
#SBATCH --time=18:00:00
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

# Fine-tune recipe = the "anchored" recipe (the strongest one we have):
#   - encoder LR 6e-4 at top, layer_decay 0.85 (preserve low-level SSL features)
#   - head LR 1e-3
#   - no encoder freeze, 5-epoch rewarmup
#   - InterCTC at layer 7, weight 0.15
#   - SpecAug fills with the SSL learned mask_emb (matches pretrain distribution)
#   - SSL anchor: K=100 + K=500 CE @ 0.1, heads warm-started from SSL pretraining
#
# IDENTICAL across all three codebook conditions — the only variable is the
# SSL pretraining objective. Comparison is then "given different SSL-trained
# encoders, what's the best downstream WER under the same fine-tune recipe."
ENCODER_LR="${ENCODER_LR:-6e-4}"
HEAD_LR="${HEAD_LR:-1e-3}"
BASE_LR="${BASE_LR:-1e-3}"
ENCODER_LAYER_LR_DECAY="${ENCODER_LAYER_LR_DECAY:-0.85}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
PEAK_EPOCHS="${PEAK_EPOCHS:-50}"
NOAM_DECAY_RATE="${NOAM_DECAY_RATE:-0.5}"
FREEZE_ENCODER_EPOCHS="${FREEZE_ENCODER_EPOCHS:-0}"
ENCODER_REWARMUP_EPOCHS="${ENCODER_REWARMUP_EPOCHS:-5}"
INTER_CTC_LAYERS="${INTER_CTC_LAYERS:-7}"
INTER_CTC_WEIGHT="${INTER_CTC_WEIGHT:-0.15}"
SPECAUG_MASK_SOURCE="${SPECAUG_MASK_SOURCE:-ssl-mask}"
SSL_ANCHOR_WEIGHT="${SSL_ANCHOR_WEIGHT:-0.1}"
SSL_ANCHOR_TARGETS_DIR="${SSL_ANCHOR_TARGETS_DIR:-${TARGETS_DIR}}"
SSL_ANCHOR_LOAD_HEADS="${SSL_ANCHOR_LOAD_HEADS:-1}"
NO_DECAY_NORM_BIAS="${NO_DECAY_NORM_BIAS:-1}"

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

read -r -a INTER_CTC_LAYERS_ARGS <<< "${INTER_CTC_LAYERS}"
EXTRA_ARGS=(
    --encoder-layer-lr-decay "${ENCODER_LAYER_LR_DECAY}"
    --specaug-mask-source "${SPECAUG_MASK_SOURCE}"
    --freeze-encoder-epochs "${FREEZE_ENCODER_EPOCHS}"
    --encoder-rewarmup-epochs "${ENCODER_REWARMUP_EPOCHS}"
    --inter-ctc-layers "${INTER_CTC_LAYERS_ARGS[@]}"
    --inter-ctc-weight "${INTER_CTC_WEIGHT}"
    --ssl-anchor-weight "${SSL_ANCHOR_WEIGHT}"
    --ssl-anchor-targets-dir "${SSL_ANCHOR_TARGETS_DIR}"
)
if [ "${NO_DECAY_NORM_BIAS}" = "1" ]; then
    EXTRA_ARGS+=(--no-decay-norm-and-bias)
fi
if [ "${SSL_ANCHOR_LOAD_HEADS}" = "1" ]; then
    EXTRA_ARGS+=(--ssl-anchor-load-heads)
fi

echo "Job ${SLURM_JOB_ID} | TRAIN_HOURS=${TRAIN_HOURS} | EPOCHS=${EPOCHS}"
echo "SSL_CHECKPOINT: ${SSL_CHECKPOINT}"
echo "FT_OUTPUT_DIR:  ${FT_OUTPUT_DIR}"
echo "Effective batch: $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
echo "Recipe: anchored — LR ${ENCODER_LR}/decay ${ENCODER_LAYER_LR_DECAY}, freeze=${FREEZE_ENCODER_EPOCHS}+rewarmup=${ENCODER_REWARMUP_EPOCHS}, InterCTC@${INTER_CTC_LAYERS}/${INTER_CTC_WEIGHT}, ssl-anchor=${SSL_ANCHOR_WEIGHT} (load_heads=${SSL_ANCHOR_LOAD_HEADS}), specaug_src=${SPECAUG_MASK_SOURCE}"

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
    "${EXTRA_ARGS[@]}" \
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
