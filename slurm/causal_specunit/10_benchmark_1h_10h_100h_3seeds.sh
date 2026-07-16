#!/bin/bash
# One Slurm job for the full benchmark:
#   3 data variants x 3 models x 3 seeds = 27 fine-tunes.
#
# Data variants:
#   librilight_1h, librilight_10h, train-clean-100
#
# Models:
#   scratch, iter1, iter2, distill_hubert, distill_wav2vec2
#     (distill_* = reviewer-requested KD baseline: HuBERT/wav2vec2 Base distilled
#      into the SAME 9M SqueezeFormer-XS. Same fine-tune recipe as iter1/iter2,
#      only --ssl-checkpoint differs, so the rows are apples-to-apples.)
#
# Seeds:
#   42, 43, 44
#
# Submit:
#   sbatch slurm/causal_specunit/10_benchmark_1h_10h_100h_3seeds.sh
#
# Fine-tune ONLY the distilled models at 1h (matches the recipe/seeds/output tree
# already used for scratch/iter1/iter2, so the numbers drop straight into the table):
#   SUBSETS=librilight_1h CONDITIONS="distill_hubert distill_wav2vec2" \
#       sbatch slurm/causal_specunit/10_benchmark_1h_10h_100h_3seeds.sh
#
# Optional filters:
#   SUBSETS="librilight_1h librilight_10h" sbatch ...
#   CONDITIONS="scratch iter1 iter2 distill_hubert distill_wav2vec2" sbatch ...
#   SEED_LIST="42" sbatch ...
#   CLEAN_FIRST=1 sbatch ...
#   DISTILL_HUBERT_CKPT=... DISTILL_W2V2_CKPT=... sbatch ...   # override ckpt paths
#
# Shared recipe across all cells. Only per-GPU batch size changes by split.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_asr_bench
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_asr_bench.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_asr_bench.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-dataset/bpe128.model}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/causal_specunit/benchmark_1h_10h_100h_4gpu}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

if [ -n "${SUBSETS:-}" ]; then
    read -r -a DATASETS <<< "${SUBSETS}"
else
    DATASETS=(librilight_1h librilight_10h train-clean-100)
fi

if [ -n "${CONDITIONS:-}" ]; then
    read -r -a MODEL_CONDITIONS <<< "${CONDITIONS}"
else
    MODEL_CONDITIONS=(scratch iter1 iter2)
fi

if [ -n "${SEED_LIST:-}" ]; then
    read -r -a SEEDS <<< "${SEED_LIST}"
else
    SEEDS=(42 43 44)
fi

# Shared recipe. Do not specialize by split except BATCH_SIZE in
# batch_size_for_split().
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
mkdir -p logs

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

resolve_ssl_ckpt() {
    case "$1" in
        scratch)          echo "" ;;
        iter1)            echo "outputs/causal_specunit/pretrain_ssl_v2_150k_c8/checkpoint_step150000/checkpoint.pt" ;;
        iter2)            echo "outputs/causal_specunit/pretrain_ssl_iter2_from_v2_c8/checkpoint_step100000/checkpoint.pt" ;;
        # KD baseline: distilled 9M SqueezeFormer-XS students (250k steps). The
        # distilled encoder saves under encoder.* so train_ctc loads it exactly
        # like any SSL checkpoint. Override paths via DISTILL_HUBERT_CKPT / DISTILL_W2V2_CKPT.
        distill_hubert)   echo "${DISTILL_HUBERT_CKPT:-outputs/causal_specunit/distill_hubert_base_960h/checkpoint_step250000/checkpoint.pt}" ;;
        distill_wav2vec2) echo "${DISTILL_W2V2_CKPT:-outputs/causal_specunit/distill_wav2vec2_base_960h/checkpoint_step250000/checkpoint.pt}" ;;
        *)                echo "ERROR"; return 1 ;;
    esac
}

# Sanity checks
[ -d "${VIRTUAL_ENV}" ]          || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}/dev-other" ]  || { echo "Missing eval split: ${DATA_ROOT}/dev-other"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]  || { echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]       || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }

for DATASET in "${DATASETS[@]}"; do
    case "${DATASET}" in
        librilight_1h|librilight_10h|train-clean-100) ;;
        *) echo "Invalid split ${DATASET}; expected librilight_1h, librilight_10h, or train-clean-100"; exit 1 ;;
    esac
    [ -d "${DATA_ROOT}/${DATASET}" ] || { echo "Missing train split: ${DATA_ROOT}/${DATASET}"; exit 1; }
done

for CONDITION in "${MODEL_CONDITIONS[@]}"; do
    SSL_CHECKPOINT=$(resolve_ssl_ckpt "${CONDITION}")
    if [ "${SSL_CHECKPOINT}" = "ERROR" ]; then
        echo "Invalid condition ${CONDITION}; expected scratch, iter1, iter2, distill_hubert, or distill_wav2vec2"
        exit 1
    fi
    if [ -n "${SSL_CHECKPOINT}" ] && [ ! -f "${SSL_CHECKPOINT}" ]; then
        echo "Missing SSL checkpoint for ${CONDITION}: ${SSL_CHECKPOINT}"
        exit 1
    fi
done

PORT_BASE=$((31000 + (SLURM_JOB_ID % 3000)))
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

TOTAL_CELLS=$((${#DATASETS[@]} * ${#MODEL_CONDITIONS[@]} * ${#SEEDS[@]}))
CELL_IDX=0

log_phase "START variants=${DATASETS[*]} models=${MODEL_CONDITIONS[*]} seeds=${SEEDS[*]} nproc=${NPROC_PER_NODE} total_cells=${TOTAL_CELLS}"

for SUBSET in "${DATASETS[@]}"; do
    BATCH_SIZE=$(batch_size_for_split "${SUBSET}")

    for CONDITION in "${MODEL_CONDITIONS[@]}"; do
        SSL_CHECKPOINT=$(resolve_ssl_ckpt "${CONDITION}")
        SSL_ARGS=()
        if [ -n "${SSL_CHECKPOINT}" ]; then
            SSL_ARGS=(--ssl-checkpoint "${SSL_CHECKPOINT}")
        fi

        for SEED in "${SEEDS[@]}"; do
            OUT_DIR="${OUTPUT_ROOT}/${SUBSET}/${CONDITION}_seed${SEED}"
            mkdir -p "${OUT_DIR}"

            log_phase "CELL ${CELL_IDX}/${TOTAL_CELLS}: variant=${SUBSET} model=${CONDITION} seed=${SEED} batch=${BATCH_SIZE}"

            if [ "${CLEAN_FIRST:-0}" = "1" ] && [ -d "${OUT_DIR}" ]; then
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
                    ${SSL_ARGS[@]+"${SSL_ARGS[@]}"} \
                    --output-dir "${OUT_DIR}" \
                    --variant xs \
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
                --variant xs \
                --splits test-clean test-other \
                --batch-size "${EVAL_TEST_BATCH_SIZE}" \
                --workers 4 \
                --output "${OUT_DIR}/eval_results.json"

            CELL_IDX=$((CELL_IDX + 1))
        done
    done
done

log_phase "DONE all ${TOTAL_CELLS} cells"
echo "Results:"
for SUBSET in "${DATASETS[@]}"; do
    for CONDITION in "${MODEL_CONDITIONS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo "  ${OUTPUT_ROOT}/${SUBSET}/${CONDITION}_seed${SEED}/eval_results.json"
        done
    done
done
