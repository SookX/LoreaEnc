#!/bin/bash
# MelHuBERT-style Transformer baseline at the compact (~9M) scale.
#
# This is a related-work baseline, not an ablation of SqueezeFormer:
#   1. pretrain a mel-input Transformer encoder (variant=mh9m)
#      with a single learned K-means codebook target;
#   2. fine-tune the pretrained encoder on Libri-Light 1h/10h and
#      LibriSpeech train-clean-100 with 3 seeds;
#   3. evaluate each cell on test-clean/test-other.
#
# Submit:
#   sbatch slurm/causal_specunit/12_melhubert_transformer_mh9m_baseline.sh
#
# Useful filters:
#   SEED_LIST="42" SUBSETS="librilight_10h" sbatch ...
#   MAX_STEPS=50000 sbatch ...   # smoke/proof-of-concept pretrain

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

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-dataset/bpe128.model}"
VARIANT="${VARIANT:-mh9m}"
CODEBOOK_MODE="${CODEBOOK_MODE:-fine}"
MAX_STEPS="${MAX_STEPS:-150000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PRETRAIN_NPROC_PER_NODE="${PRETRAIN_NPROC_PER_NODE:-${NPROC_PER_NODE}}"
FT_NPROC_PER_NODE="${FT_NPROC_PER_NODE:-${NPROC_PER_NODE}}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-${FT_NPROC_PER_NODE}}"

SSL_OUTPUT_DIR="${SSL_OUTPUT_DIR:-outputs/causal_specunit/melhubert_transformer_mh9m/ssl_${CODEBOOK_MODE}_${MAX_STEPS}}"
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

PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-64}"
PRETRAIN_GRAD_ACCUM_STEPS="${PRETRAIN_GRAD_ACCUM_STEPS:-1}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
PRETRAIN_MASK_PROB="${PRETRAIN_MASK_PROB:-0.80}"
PRETRAIN_MASK_LENGTH="${PRETRAIN_MASK_LENGTH:-10}"
PRETRAIN_WORKERS="${PRETRAIN_WORKERS:-8}"

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
SPECAUG_DISABLE_LAST_EPOCHS="${SPECAUG_DISABLE_LAST_EPOCHS:-10}"
SAVE_EVERY="${SAVE_EVERY:-0}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-1}"

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

batch_size_for_split() {
    case "$1" in
        librilight_1h)   echo "${BATCH_SIZE_1H:-8}" ;;
        librilight_10h)  echo "${BATCH_SIZE_10H:-32}" ;;
        train-clean-100) echo "${BATCH_SIZE_100H:-128}" ;;
        *) echo "ERROR"; return 1 ;;
    esac
}

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

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
PORT_BASE=$((32000 + (SLURM_JOB_ID % 2500)))

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

TOTAL_CELLS=$((${#DATASETS[@]} * ${#SEEDS[@]}))
CELL_IDX=0

log_phase "FINE-TUNE ${VARIANT} baseline subsets=${DATASETS[*]} seeds=${SEEDS[*]} total_cells=${TOTAL_CELLS}"
echo "Processes: pretrain=${PRETRAIN_NPROC_PER_NODE} finetune=${FT_NPROC_PER_NODE} eval=${EVAL_NPROC_PER_NODE}"

for SUBSET in "${DATASETS[@]}"; do
    BATCH_SIZE=$(batch_size_for_split "${SUBSET}")

    for SEED in "${SEEDS[@]}"; do
        OUT_DIR="${OUTPUT_ROOT}/${SUBSET}/melhubert_tx_${CODEBOOK_MODE}_seed${SEED}"
        TRAIN_DONE="${OUT_DIR}/train.done"
        mkdir -p "${OUT_DIR}"

        log_phase "CELL ${CELL_IDX}/${TOTAL_CELLS}: subset=${SUBSET} seed=${SEED} batch=${BATCH_SIZE}"

        if [ -f "${OUT_DIR}/eval_results.json" ]; then
            echo "SKIP cell: eval_results.json exists at ${OUT_DIR}/eval_results.json"
            CELL_IDX=$((CELL_IDX + 1))
            continue
        fi

        TRAIN_PORT=$((PORT_BASE + 10 + CELL_IDX * 2))
        EVAL_PORT=$((PORT_BASE + 11 + CELL_IDX * 2))

        if [ ! -f "${OUT_DIR}/checkpoint_best/checkpoint.pt" ] || [ ! -f "${TRAIN_DONE}" ]; then
            rm -f "${TRAIN_DONE}"
            torchrun \
                --nproc_per_node="${FT_NPROC_PER_NODE}" \
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
                --workers 8 \
                --dataloader-timeout 120 \
                --lr "${BASE_LR}" \
                --encoder-lr "${ENCODER_LR}" \
                --head-lr "${HEAD_LR}" \
                --warmup-epochs "${WARMUP_EPOCHS}" \
                --peak-epochs "${PEAK_EPOCHS}" \
                --noam-decay-rate "${NOAM_DECAY_RATE}" \
                --max-grad-norm "${MAX_GRAD_NORM}" \
                --specaug \
                --specaug-time-mask-param 30 \
                --specaug-freq-mask-param 20 \
                --specaug-time-masks 2 \
                --specaug-freq-masks 2 \
                --specaug-disable-last-epochs "${SPECAUG_DISABLE_LAST_EPOCHS}" \
                --seed "${SEED}" \
                --progress off \
                --log-every 0 \
                --save-every "${SAVE_EVERY}" \
                --keep-checkpoints "${KEEP_CHECKPOINTS}"
            touch "${TRAIN_DONE}"
        else
            echo "SKIP train: train.done and checkpoint_best exist at ${OUT_DIR}"
        fi

        torchrun \
            --nproc_per_node="${EVAL_NPROC_PER_NODE}" \
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

log_phase "DONE ${VARIANT} MelHuBERT-style Transformer baseline"
echo "SSL checkpoint: ${SSL_CKPT}"
echo "Results root:   ${OUTPUT_ROOT}"
