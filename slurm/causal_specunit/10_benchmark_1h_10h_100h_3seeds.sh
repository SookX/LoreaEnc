#!/bin/bash
# Benchmark ASR fine-tunes for:
#   - Libri-Light 1h
#   - Libri-Light 10h
#   - LibriSpeech train-clean-100
#
# Array layout: 3 splits x 3 initializations x 3 seeds = 27 jobs.
#
# Submit the full benchmark:
#   sbatch --array=0-26 slurm/causal_specunit/10_benchmark_1h_10h_100h_3seeds.sh
#
# Or run one explicit cell:
#   sbatch --export=ALL,SUBSET=librilight_1h,CONDITION=iter1,SEED=42 \
#     slurm/causal_specunit/10_benchmark_1h_10h_100h_3seeds.sh
#
# Only the per-GPU batch size changes by split. Optimizer, schedule,
# SpecAugment, epochs, eval, and checkpoint settings are shared.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_asr_bench
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_asr_bench.%A_%a.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_asr_bench.%A_%a.err

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

SUBSETS_DEFAULT=(librilight_1h librilight_10h train-clean-100)
CONDITIONS_DEFAULT=(scratch iter1 iter2)
SEEDS_DEFAULT=(42 43 44)

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    NUM_SEEDS="${#SEEDS_DEFAULT[@]}"
    NUM_CONDITIONS="${#CONDITIONS_DEFAULT[@]}"
    SEED_INDEX=$((TASK_ID % NUM_SEEDS))
    CONDITION_INDEX=$(((TASK_ID / NUM_SEEDS) % NUM_CONDITIONS))
    SUBSET_INDEX=$((TASK_ID / (NUM_SEEDS * NUM_CONDITIONS)))

    if [ "${SUBSET_INDEX}" -ge "${#SUBSETS_DEFAULT[@]}" ]; then
        echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0-26"
        exit 1
    fi

    SUBSET="${SUBSETS_DEFAULT[${SUBSET_INDEX}]}"
    CONDITION="${CONDITIONS_DEFAULT[${CONDITION_INDEX}]}"
    SEED="${SEEDS_DEFAULT[${SEED_INDEX}]}"
elif [ -n "${SUBSET:-}" ] || [ -n "${CONDITION:-}" ] || [ -n "${SEED:-}" ]; then
    : "${SUBSET:?Required when running an explicit cell}"
    : "${CONDITION:?Required when running an explicit cell}"
    : "${SEED:?Required when running an explicit cell}"
else
    SUBSET="${SUBSETS_DEFAULT[0]}"
    CONDITION="${CONDITIONS_DEFAULT[0]}"
    SEED="${SEEDS_DEFAULT[0]}"
fi

case "${SUBSET}" in
    librilight_1h)
        BATCH_SIZE="${BATCH_SIZE_1H:-8}"
        ;;
    librilight_10h)
        BATCH_SIZE="${BATCH_SIZE_10H:-32}"
        ;;
    train-clean-100)
        BATCH_SIZE="${BATCH_SIZE_100H:-128}"
        ;;
    *)
        echo "Invalid SUBSET=${SUBSET}; expected librilight_1h, librilight_10h, or train-clean-100"
        exit 1
        ;;
esac

case "${CONDITION}" in
    scratch) SSL_CHECKPOINT="" ;;
    iter1)   SSL_CHECKPOINT="outputs/causal_specunit/pretrain_ssl_v2_150k_c8/checkpoint_step150000/checkpoint.pt" ;;
    iter2)   SSL_CHECKPOINT="outputs/causal_specunit/pretrain_ssl_iter2_from_v2_c8/checkpoint_step100000/checkpoint.pt" ;;
    *)       echo "Invalid CONDITION=${CONDITION}; expected scratch, iter1, or iter2"; exit 1 ;;
esac

# Shared recipe. Do not specialize by split except BATCH_SIZE above.
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

OUT_DIR="${OUTPUT_ROOT}/${SUBSET}/${CONDITION}_seed${SEED}"

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
mkdir -p logs "${OUT_DIR}"

[ -d "${VIRTUAL_ENV}" ]          || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}/${SUBSET}" ]  || { echo "Missing train split: ${DATA_ROOT}/${SUBSET}"; exit 1; }
[ -d "${DATA_ROOT}/dev-other" ]  || { echo "Missing eval split: ${DATA_ROOT}/dev-other"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]  || { echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]       || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }
if [ -n "${SSL_CHECKPOINT}" ] && [ ! -f "${SSL_CHECKPOINT}" ]; then
    echo "Missing SSL checkpoint for ${CONDITION}: ${SSL_CHECKPOINT}"
    exit 1
fi

SSL_ARGS=()
if [ -n "${SSL_CHECKPOINT}" ]; then
    SSL_ARGS=(--ssl-checkpoint "${SSL_CHECKPOINT}")
fi

PORT_BASE=$((31000 + (SLURM_JOB_ID % 4000)))
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

echo "Benchmark cell:"
echo "  subset=${SUBSET}"
echo "  condition=${CONDITION}"
echo "  seed=${SEED}"
echo "  output=${OUT_DIR}"
echo "  nproc=${NPROC_PER_NODE}"
echo "  epochs=${EPOCHS}"
echo "  batch_size=${BATCH_SIZE}"
echo "  grad_accum=${GRAD_ACCUM_STEPS}"
echo "  lr=${BASE_LR} encoder_lr=${ENCODER_LR} head_lr=${HEAD_LR}"

if [ "${CLEAN_FIRST:-0}" = "1" ] && [ -d "${OUT_DIR}" ]; then
    find "${OUT_DIR}" -mindepth 1 ! -name ".nfs*" -delete 2>/dev/null || true
fi

if [ ! -f "${OUT_DIR}/checkpoint_best/checkpoint.pt" ]; then
    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${PORT_BASE}" \
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

if [ ! -f "${OUT_DIR}/eval_results.json" ]; then
    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="$((PORT_BASE + 1))" \
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
else
    echo "SKIP eval: eval_results.json exists at ${OUT_DIR}/eval_results.json"
fi

echo "DONE subset=${SUBSET} condition=${CONDITION} seed=${SEED} output=${OUT_DIR}"
