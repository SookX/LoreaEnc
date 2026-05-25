#!/bin/bash
# MelHuBERT-style Transformer baseline at the compact (~9M) scale.
#
# This is a related-work baseline. It first pretrains a mel-input
# Transformer encoder (variant=mh9m) with a single learned K-means
# codebook target, then fine-tunes on the same 1h/10h/100h labeled
# splits using the EXACT SAME downstream CTC recipe as
# 10_benchmark_1h_10h_100h_3seeds.sh, so the resulting numbers are
# directly comparable to the iter-1/iter-2 rows in the benchmark table.
#
# The only intended differences from script 10 are:
#   1. A leading SSL-pretrain stage that produces the baseline encoder.
#   2. --variant mh9m (Transformer) instead of --variant xs (SqueezeFormer).
#   3. A single model row instead of a scratch/iter1/iter2 loop.
# Every fine-tune hyperparameter (epochs, batch-size-per-split, LRs,
# warmup/peak schedule, SpecAugment) is inherited from script 10's globals.
#
# Submit:
#   sbatch slurm/causal_specunit/12_melhubert_transformer_mh9m_baseline.sh
#
# Optional filters (same semantics as script 10):
#   SUBSETS="librilight_1h librilight_10h" sbatch ...
#   SEED_LIST="42" sbatch ...
#   CLEAN_FIRST=1 sbatch ...
#   FORCE_TRAIN=1 sbatch ...      # alias for CLEAN_FIRST=1
#
# Pretrain-specific:
#   MAX_STEPS=50000 sbatch ...    # shorter pretrain (smoke test)
#   CODEBOOK_MODE=fine sbatch ... # MelHuBERT-style single fine codebook

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_mh9m
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_mh9m.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_mh9m.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

# ----------------------------------------------------------------------
# Paths and shared defaults (same as script 10)
# ----------------------------------------------------------------------
PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-dataset/bpe128.model}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# ----------------------------------------------------------------------
# MelHuBERT pretrain-specific
# ----------------------------------------------------------------------
VARIANT="${VARIANT:-mh9m}"
CODEBOOK_MODE="${CODEBOOK_MODE:-fine}"
MAX_STEPS="${MAX_STEPS:-150000}"
PRETRAIN_NPROC_PER_NODE="${PRETRAIN_NPROC_PER_NODE:-${NPROC_PER_NODE}}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-64}"
PRETRAIN_GRAD_ACCUM_STEPS="${PRETRAIN_GRAD_ACCUM_STEPS:-1}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
PRETRAIN_MASK_PROB="${PRETRAIN_MASK_PROB:-0.80}"
PRETRAIN_MASK_LENGTH="${PRETRAIN_MASK_LENGTH:-10}"
PRETRAIN_WORKERS="${PRETRAIN_WORKERS:-8}"
SSL_OUTPUT_DIR="${SSL_OUTPUT_DIR:-outputs/causal_specunit/melhubert_transformer_mh9m/ssl_${CODEBOOK_MODE}_${MAX_STEPS}}"

# ----------------------------------------------------------------------
# Fine-tune output root and selectors
# ----------------------------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/causal_specunit/melhubert_transformer_mh9m/benchmark_${CODEBOOK_MODE}_${MAX_STEPS}}"

if [ -n "${SUBSETS:-}" ]; then
    read -r -a DATASETS <<< "${SUBSETS}"
else
    DATASETS=(librilight_1h librilight_10h train-clean-100)
fi

if [ -n "${SEED_LIST:-}" ]; then
    read -r -a SEEDS <<< "${SEED_LIST}"
else
    SEEDS=(42 43 44)
fi

# ----------------------------------------------------------------------
# Fine-tune recipe globals (COPIED VERBATIM from script 10).
# Do not specialise by split except BATCH_SIZE via batch_size_for_split().
# ----------------------------------------------------------------------
EPOCHS="${EPOCHS:-150}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_TEST_BATCH_SIZE="${EVAL_TEST_BATCH_SIZE:-64}"
BASE_LR="${BASE_LR:-1e-3}"
ENCODER_LR="${ENCODER_LR:-3e-4}"
HEAD_LR="${HEAD_LR:-1e-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
PEAK_EPOCHS="${PEAK_EPOCHS:-50}"
NOAM_DECAY_RATE="${NOAM_DECAY_RATE:-0.5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SPECAUG_TIME_MASK_PARAM="${SPECAUG_TIME_MASK_PARAM:-30}"
SPECAUG_FREQ_MASK_PARAM="${SPECAUG_FREQ_MASK_PARAM:-20}"
SPECAUG_TIME_MASKS="${SPECAUG_TIME_MASKS:-2}"
SPECAUG_FREQ_MASKS="${SPECAUG_FREQ_MASKS:-2}"
SPECAUG_DISABLE_LAST_EPOCHS="${SPECAUG_DISABLE_LAST_EPOCHS:-10}"
SAVE_EVERY="${SAVE_EVERY:-10}"
WORKERS="${WORKERS:-8}"
DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-120}"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "${PROJECT_DIR}"
mkdir -p logs "${SSL_OUTPUT_DIR}" "${OUTPUT_ROOT}"

log_phase() {
    echo ""
    echo "===================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "===================================================="
}

# IDENTICAL to script 10.
batch_size_for_split() {
    case "$1" in
        librilight_1h)   echo "${BATCH_SIZE_1H:-8}" ;;
        librilight_10h)  echo "${BATCH_SIZE_10H:-32}" ;;
        train-clean-100) echo "${BATCH_SIZE_100H:-128}" ;;
        *) echo "ERROR"; return 1 ;;
    esac
}

# ----------------------------------------------------------------------
# Sanity checks. cmvn.pt + tokenizer + dev-other match script 10;
# targets.pt is additionally required because we run an SSL pretrain
# stage (pretrain_ssl.py loads ${TARGETS_DIR}/targets.pt directly).
# ----------------------------------------------------------------------
[ -d "${VIRTUAL_ENV}" ]            || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}/dev-other" ]    || { echo "Missing eval split: ${DATA_ROOT}/dev-other"; exit 1; }
[ -f "${TARGETS_DIR}/targets.pt" ] || { echo "Missing targets: ${TARGETS_DIR}/targets.pt"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]    || { echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]         || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }

for DATASET in "${DATASETS[@]}"; do
    case "${DATASET}" in
        librilight_1h|librilight_10h|train-clean-100) ;;
        *) echo "Invalid split ${DATASET}; expected librilight_1h, librilight_10h, or train-clean-100"; exit 1 ;;
    esac
    [ -d "${DATA_ROOT}/${DATASET}" ] || { echo "Missing train split: ${DATA_ROOT}/${DATASET}"; exit 1; }
done

PORT_BASE=$((32000 + (SLURM_JOB_ID % 2500)))
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# ----------------------------------------------------------------------
# Stage 1: SSL pretrain (MelHuBERT-style Transformer, single codebook)
# ----------------------------------------------------------------------
SSL_CKPT="${SSL_OUTPUT_DIR}/checkpoint_step$(printf '%06d' "${MAX_STEPS}")/checkpoint.pt"
if [ -f "${SSL_CKPT}" ]; then
    log_phase "SKIP SSL pretrain: checkpoint exists at ${SSL_CKPT}"
else
    log_phase "PRETRAIN ${VARIANT} Transformer | codebook=${CODEBOOK_MODE} | steps=${MAX_STEPS}"
    torchrun \
        --nproc_per_node="${PRETRAIN_NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="$((PORT_BASE + 1))" \
        -m CausalSpecUnit.pretrain_ssl \
        --data-root "${DATA_ROOT}" \
        --targets-dir "${TARGETS_DIR}" \
        --output-dir "${SSL_OUTPUT_DIR}" \
        --variant "${VARIANT}" \
        --epochs 1000 \
        --max-steps "${MAX_STEPS}" \
        --batch-size "${PRETRAIN_BATCH_SIZE}" \
        --grad-accum-steps "${PRETRAIN_GRAD_ACCUM_STEPS}" \
        --codebook-mode "${CODEBOOK_MODE}" \
        --mask-prob "${PRETRAIN_MASK_PROB}" \
        --mask-length "${PRETRAIN_MASK_LENGTH}" \
        --chunk-size 8 \
        --chunk-stride 4 \
        --lr "${PRETRAIN_LR}" \
        --warmup-epochs 10 \
        --peak-epochs 10 \
        --noam-decay-rate 1.0 \
        --max-grad-norm 1.0 \
        --max-safe-grad-norm 200.0 \
        --workers "${PRETRAIN_WORKERS}" \
        --dataloader-timeout 300 \
        --prefetch-factor 4 \
        --log-every 100 \
        --save-every 5 \
        --progress off
fi

[ -f "${SSL_CKPT}" ] || { echo "Missing SSL checkpoint after pretrain: ${SSL_CKPT}"; exit 1; }

# ----------------------------------------------------------------------
# Stage 2: fine-tune loop. STRUCTURE IDENTICAL TO SCRIPT 10's main loop,
# minus the MODEL_CONDITIONS dimension (we have one model: MelHuBERT-tx).
# ----------------------------------------------------------------------
TOTAL_CELLS=$((${#DATASETS[@]} * ${#SEEDS[@]}))
CELL_IDX=0

log_phase "START variants=${DATASETS[*]} seeds=${SEEDS[*]} nproc=${NPROC_PER_NODE} total_cells=${TOTAL_CELLS}"

for SUBSET in "${DATASETS[@]}"; do
    BATCH_SIZE=$(batch_size_for_split "${SUBSET}")

    for SEED in "${SEEDS[@]}"; do
        OUT_DIR="${OUTPUT_ROOT}/${SUBSET}/melhubert_tx_${CODEBOOK_MODE}_seed${SEED}"
        mkdir -p "${OUT_DIR}"

        log_phase "CELL ${CELL_IDX}/${TOTAL_CELLS}: variant=${SUBSET} model=melhubert_tx_${CODEBOOK_MODE} seed=${SEED} batch=${BATCH_SIZE}"

        # FORCE_TRAIN=1 is an alias for CLEAN_FIRST=1 (script 10 uses CLEAN_FIRST only).
        if { [ "${CLEAN_FIRST:-0}" = "1" ] || [ "${FORCE_TRAIN:-0}" = "1" ]; } && [ -d "${OUT_DIR}" ]; then
            find "${OUT_DIR}" -mindepth 1 ! -name ".nfs*" -delete 2>/dev/null || true
        fi

        if [ -f "${OUT_DIR}/eval_results.json" ]; then
            echo "SKIP cell: eval_results.json exists at ${OUT_DIR}/eval_results.json"
            CELL_IDX=$((CELL_IDX + 1))
            continue
        fi

        TRAIN_PORT=$((PORT_BASE + CELL_IDX * 2 + 1))
        EVAL_PORT=$((PORT_BASE + CELL_IDX * 2 + 2))

        if [ ! -f "${OUT_DIR}/checkpoint_best/checkpoint.pt" ]; then
            torchrun \
                --nproc_per_node="${NPROC_PER_NODE}" \
                --master_addr="${MASTER_ADDR}" \
                --master_port="${TRAIN_PORT}" \
                -m CausalSpecUnit.train_ctc \
                --data-root "${DATA_ROOT}" \
                --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
                --tokenizer-path "${TOKENIZER_PATH}" \
                --train-splits "${SUBSET}" \
                --ssl-checkpoint "${SSL_CKPT}" \
                --output-dir "${OUT_DIR}" \
                --variant "${VARIANT}" \
                --epochs "${EPOCHS}" \
                --batch-size "${BATCH_SIZE}" \
                --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
                --eval-batch-size "${EVAL_BATCH_SIZE}" \
                --eval-split dev-other \
                --eval-every 1 \
                --workers "${WORKERS}" \
                --dataloader-timeout "${DATALOADER_TIMEOUT}" \
                --lr "${BASE_LR}" \
                --encoder-lr "${ENCODER_LR}" \
                --head-lr "${HEAD_LR}" \
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
                --seed "${SEED}" \
                --progress off \
                --log-every 0 \
                --save-every "${SAVE_EVERY}"
        else
            echo "SKIP train: checkpoint_best exists at ${OUT_DIR}/checkpoint_best/checkpoint.pt"
        fi

        torchrun \
            --nproc_per_node="${NPROC_PER_NODE}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${EVAL_PORT}" \
            -m CausalSpecUnit.evaluate_ctc \
            --checkpoint "${OUT_DIR}/checkpoint_best/checkpoint.pt" \
            --data-root "${DATA_ROOT}" \
            --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
            --tokenizer-path "${TOKENIZER_PATH}" \
            --variant "${VARIANT}" \
            --splits test-clean test-other \
            --batch-size "${EVAL_TEST_BATCH_SIZE}" \
            --workers 4 \
            --output "${OUT_DIR}/eval_results.json"

        CELL_IDX=$((CELL_IDX + 1))
    done
done

log_phase "DONE ${VARIANT} MelHuBERT-style Transformer baseline (${TOTAL_CELLS} cells)"
echo "SSL checkpoint: ${SSL_CKPT}"
echo "Results root:   ${OUTPUT_ROOT}"
for SUBSET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "  ${OUTPUT_ROOT}/${SUBSET}/melhubert_tx_${CODEBOOK_MODE}_seed${SEED}/eval_results.json"
    done
done
