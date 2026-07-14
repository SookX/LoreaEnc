#!/bin/bash
# Fine-tuning matrix on official Libri-Light limited-supervision splits.
#
# Array layout: 3 official subsets x 3 initializations x 3 seeds = 27 jobs.
# Submit:
#   sbatch --array=0-26 slurm/causal_specunit/08_librilight_finetune_matrix.sh
#
# Prepared splits expected under DATA_ROOT:
#   librilight_10min, librilight_1h, librilight_10h

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ll_ft
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ll_ft.%A_%a.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ll_ft.%A_%a.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-dataset/bpe128.model}"

SUBSETS=(librilight_10min librilight_1h librilight_10h)
CONDITIONS=(scratch iter1 iter2)
SEEDS=(42 43 44)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SEEDS="${#SEEDS[@]}"
NUM_CONDITIONS="${#CONDITIONS[@]}"

SEED_INDEX=$((TASK_ID % NUM_SEEDS))
CONDITION_INDEX=$(((TASK_ID / NUM_SEEDS) % NUM_CONDITIONS))
SUBSET_INDEX=$((TASK_ID / (NUM_SEEDS * NUM_CONDITIONS)))

if [ "${SUBSET_INDEX}" -ge "${#SUBSETS[@]}" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0-26"
    exit 1
fi

SUBSET="${SUBSETS[${SUBSET_INDEX}]}"
CONDITION="${CONDITIONS[${CONDITION_INDEX}]}"
SEED="${SEEDS[${SEED_INDEX}]}"

case "${CONDITION}" in
    scratch) SSL_CHECKPOINT="" ;;
    iter1)   SSL_CHECKPOINT="outputs/causal_specunit/pretrain_ssl_v2_150k_c8/checkpoint_step150000/checkpoint.pt" ;;
    iter2)   SSL_CHECKPOINT="outputs/causal_specunit/pretrain_ssl_iter2_from_v2_c8/checkpoint_step100000/checkpoint.pt" ;;
    *)       echo "Invalid CONDITION=${CONDITION}"; exit 1 ;;
esac

OUT_DIR="outputs/causal_specunit/librilight_matrix/${SUBSET}/${CONDITION}_seed${SEED}"

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

[ -d "${VIRTUAL_ENV}" ]            || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}/${SUBSET}" ]    || { echo "Missing prepared split: ${DATA_ROOT}/${SUBSET}"; exit 1; }
[ -d "${DATA_ROOT}/dev-other" ]    || { echo "Missing eval split: ${DATA_ROOT}/dev-other"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]    || { echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]         || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }
if [ -n "${SSL_CHECKPOINT}" ] && [ ! -f "${SSL_CHECKPOINT}" ]; then
    echo "Missing SSL checkpoint for ${CONDITION}: ${SSL_CHECKPOINT}"
    exit 1
fi

SSL_ARGS=()
if [ -n "${SSL_CHECKPOINT}" ]; then
    SSL_ARGS=(--ssl-checkpoint "${SSL_CHECKPOINT}")
fi

PORT_BASE=$((28000 + (SLURM_JOB_ID % 5000)))
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

if [ ! -f "${OUT_DIR}/checkpoint_best/checkpoint.pt" ]; then
    torchrun \
        --nproc_per_node=2 \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${PORT_BASE}" \
        -m CausalSpecUnit.train_ctc \
        --data-root "${DATA_ROOT}" \
        --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --train-splits "${SUBSET}" \
        "${SSL_ARGS[@]}" \
        --output-dir "${OUT_DIR}" \
        --variant xs \
        --epochs 150 \
        --batch-size 64 \
        --grad-accum-steps 1 \
        --eval-batch-size 128 \
        --eval-split dev-other \
        --eval-every 1 \
        --workers 8 \
        --dataloader-timeout 120 \
        --lr 1e-3 \
        --encoder-lr 3e-4 \
        --head-lr 1e-3 \
        --warmup-epochs 10 \
        --peak-epochs 50 \
        --noam-decay-rate 0.5 \
        --max-grad-norm 1.0 \
        --specaug \
        --specaug-time-mask-param 30 \
        --specaug-freq-mask-param 20 \
        --specaug-time-masks 2 \
        --specaug-freq-masks 2 \
        --specaug-disable-last-epochs 10 \
        --seed "${SEED}" \
        --progress off \
        --log-every 0 \
        --save-every 10
fi

if [ ! -f "${OUT_DIR}/eval_results.json" ]; then
    torchrun \
        --nproc_per_node=2 \
        --master_addr="${MASTER_ADDR}" \
        --master_port="$((PORT_BASE + 1))" \
        -m CausalSpecUnit.evaluate_ctc \
        --checkpoint "${OUT_DIR}/checkpoint_best/checkpoint.pt" \
        --data-root "${DATA_ROOT}" \
        --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --variant xs \
        --splits test-clean test-other \
        --batch-size 64 \
        --workers 4 \
        --output "${OUT_DIR}/eval_results.json"
fi

echo "DONE subset=${SUBSET} condition=${CONDITION} seed=${SEED} output=${OUT_DIR}"
