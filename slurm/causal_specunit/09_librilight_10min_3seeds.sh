#!/bin/bash
# Single-job sweep: 10-minute Libri-Light fine-tunes, 3 conditions x 3 seeds,
# all in one slurm allocation on 2 GPUs.
#
# Layout: 9 fine-tunes total, sequential within this one job.
#   conditions: scratch, iter1, iter2
#   seeds:      42, 43, 44
#
# Each (condition, seed) cell:
#   1) Trains on librilight_10min (150 epochs)
#   2) Evals checkpoint_best on test-clean + test-other
#   3) Writes eval_results.json
#
# Idempotent: any cell whose eval_results.json already exists is skipped on resubmit.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ll_10min
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ll_10min.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ll_10min.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-dataset/bpe128.model}"

SUBSET=librilight_10min
CONDITIONS=(scratch iter1 iter2)
SEEDS=(42 43 44)

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

# Sanity
[ -d "${VIRTUAL_ENV}" ]          || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}/${SUBSET}" ]  || { echo "Missing prepared split: ${DATA_ROOT}/${SUBSET}"; exit 1; }
[ -d "${DATA_ROOT}/dev-other" ]  || { echo "Missing eval split: ${DATA_ROOT}/dev-other"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]  || { echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]       || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }

PORT_BASE=$((30000 + (SLURM_JOB_ID % 5000)))
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

log_phase() {
    echo ""
    echo "===================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "===================================================="
}

resolve_ssl_ckpt() {
    case "$1" in
        scratch) echo "" ;;
        iter1)   echo "outputs/causal_specunit/pretrain_ssl_v2_150k_c8/checkpoint_step150000/checkpoint.pt" ;;
        iter2)   echo "outputs/causal_specunit/pretrain_ssl_iter2_from_v2_c8/checkpoint_step100000/checkpoint.pt" ;;
        *)       echo "ERROR"; return 1 ;;
    esac
}

CELL_IDX=0
for CONDITION in "${CONDITIONS[@]}"; do
    SSL_CHECKPOINT=$(resolve_ssl_ckpt "${CONDITION}")
    if [ -n "${SSL_CHECKPOINT}" ] && [ ! -f "${SSL_CHECKPOINT}" ]; then
        echo "Missing SSL checkpoint for ${CONDITION}: ${SSL_CHECKPOINT}"
        exit 1
    fi
    # Build the conditional --ssl-checkpoint arg as an array. Bash arrays
    # pass cleanly via expansion at the call site, but NOT through function
    # arguments — that's why this is inlined rather than wrapped in a helper.
    SSL_ARGS=()
    if [ -n "${SSL_CHECKPOINT}" ]; then
        SSL_ARGS+=(--ssl-checkpoint "${SSL_CHECKPOINT}")
    fi

    for SEED in "${SEEDS[@]}"; do
        OUT_DIR="outputs/causal_specunit/librilight_matrix/${SUBSET}/${CONDITION}_seed${SEED}"
        mkdir -p "${OUT_DIR}"

        log_phase "CELL ${CELL_IDX}/9: ${SUBSET} | ${CONDITION} | seed=${SEED}"

        if [ -f "${OUT_DIR}/eval_results.json" ]; then
            log_phase "SKIP — eval_results.json exists at ${OUT_DIR}"
            CELL_IDX=$((CELL_IDX + 1))
            continue
        fi

        TRAIN_PORT=$((PORT_BASE + CELL_IDX * 2 + 1))
        EVAL_PORT=$((PORT_BASE + CELL_IDX * 2 + 2))

        # ---- Train (skip if checkpoint_best already exists) ----
        if [ ! -f "${OUT_DIR}/checkpoint_best/checkpoint.pt" ]; then
            log_phase "  train ${CONDITION} seed=${SEED}"
            torchrun \
                --nproc_per_node=2 \
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
                --epochs 150 \
                --batch-size 8 \
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
        else
            log_phase "  SKIP train — checkpoint_best exists"
        fi

        # ---- Eval on test-clean + test-other ----
        log_phase "  eval ${CONDITION} seed=${SEED}"
        torchrun \
            --nproc_per_node=2 \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${EVAL_PORT}" \
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

        CELL_IDX=$((CELL_IDX + 1))
    done
done

log_phase "DONE all 9 cells"
echo "Results:"
for CONDITION in "${CONDITIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "  outputs/causal_specunit/librilight_matrix/${SUBSET}/${CONDITION}_seed${SEED}/eval_results.json"
    done
done
