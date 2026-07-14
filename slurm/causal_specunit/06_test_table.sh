#!/bin/bash
# Test-set WER table — one slurm job per model (scratch / iter1 / iter2).
# Each job trains on 10h then on 100h, evals each on test-clean + test-other.
# 2 GPUs. Final outputs are the per-condition eval_results.json files.
#
# Submit with the CONDITION env var:
#   sbatch --export=ALL,CONDITION=scratch slurm/causal_specunit/06_test_table.sh
#   sbatch --export=ALL,CONDITION=iter1   slurm/causal_specunit/06_test_table.sh
#   sbatch --export=ALL,CONDITION=iter2   slurm/causal_specunit/06_test_table.sh
#
# Each job is idempotent — if eval_results.json already exists for a phase, skips it.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_test_table
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_test_table.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_test_table.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
TOKENIZER_PATH="dataset/bpe128.model"

: "${CONDITION:?Required: CONDITION in {scratch, iter1, iter2}}"

# Resolve SSL checkpoint per condition. Empty for scratch.
case "${CONDITION}" in
    scratch) SSL_CHECKPOINT="" ;;
    iter1)   SSL_CHECKPOINT="outputs/causal_specunit/pretrain_ssl_v2_150k_c8/checkpoint_step150000/checkpoint.pt" ;;
    iter2)   SSL_CHECKPOINT="outputs/causal_specunit/pretrain_ssl_iter2_from_v2_c8/checkpoint_step100000/checkpoint.pt" ;;
    *)       echo "Invalid CONDITION=${CONDITION}"; exit 1 ;;
esac

BASE_OUT="outputs/causal_specunit/test_table/${CONDITION}"
OUT_10H="${BASE_OUT}_10h"
OUT_100H="${BASE_OUT}_100h"

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
mkdir -p logs "${OUT_10H}" "${OUT_100H}"

# Sanity checks
[ -d "${VIRTUAL_ENV}" ]            || { echo "Missing venv"; exit 1; }
[ -d "${DATA_ROOT}" ]              || { echo "Missing data root"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]    || { echo "Missing CMVN"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]         || { echo "Missing tokenizer"; exit 1; }
if [ -n "${SSL_CHECKPOINT}" ] && [ ! -f "${SSL_CHECKPOINT}" ]; then
    echo "Missing SSL checkpoint required for CONDITION=${CONDITION}: ${SSL_CHECKPOINT}"
    exit 1
fi

PORT_BASE=$((23000 + SLURM_JOB_ID % 10000))
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# Simple FT recipe (identical to 03_train_ctc_150ep_fair_ssl.sh defaults)
ENCODER_LR=3e-4
HEAD_LR=1e-3
BASE_LR=1e-3
WARMUP_EPOCHS=10
PEAK_EPOCHS=50
NOAM_DECAY_RATE=0.5

# Build the optional SSL checkpoint arg
SSL_ARGS=()
if [ -n "${SSL_CHECKPOINT}" ]; then
    SSL_ARGS=(--ssl-checkpoint "${SSL_CHECKPOINT}")
fi

train_ctc() {
    local out_dir="$1"; local epochs="$2"; local batch_size="$3"
    local grad_accum="$4"; local subset_arg="$5"; local master_port="$6"
    # shellcheck disable=SC2086
    torchrun \
        --nproc_per_node=2 \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${master_port}" \
        -m CausalSpecUnit.train_ctc \
        --data-root "${DATA_ROOT}" \
        --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --train-splits train-clean-100 \
        ${subset_arg} \
        "${SSL_ARGS[@]}" \
        --output-dir "${out_dir}" \
        --variant xs \
        --epochs "${epochs}" \
        --batch-size "${batch_size}" \
        --grad-accum-steps "${grad_accum}" \
        --eval-batch-size 128 \
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
}

eval_ctc() {
    local out_dir="$1"; local master_port="$2"
    torchrun \
        --nproc_per_node=2 \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${master_port}" \
        -m CausalSpecUnit.evaluate_ctc \
        --checkpoint "${out_dir}/checkpoint_best/checkpoint.pt" \
        --data-root "${DATA_ROOT}" \
        --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --variant xs \
        --splits test-clean test-other \
        --batch-size 64 \
        --workers 4 \
        --output "${out_dir}/eval_results.json"
}

# ---- 10h: train + eval ----
if [ -f "${OUT_10H}/eval_results.json" ]; then
    echo "SKIP 10h: results exist at ${OUT_10H}/eval_results.json"
else
    if [ ! -f "${OUT_10H}/checkpoint_best/checkpoint.pt" ]; then
        train_ctc "${OUT_10H}" 150 64 1 "--train-subset-hours 10 --train-subset-seed 42" $((PORT_BASE + 10))
    fi
    eval_ctc "${OUT_10H}" $((PORT_BASE + 11))
fi

# ---- 100h: train + eval ----
if [ -f "${OUT_100H}/eval_results.json" ]; then
    echo "SKIP 100h: results exist at ${OUT_100H}/eval_results.json"
else
    if [ ! -f "${OUT_100H}/checkpoint_best/checkpoint.pt" ]; then
        train_ctc "${OUT_100H}" 100 128 2 "" $((PORT_BASE + 20))
    fi
    eval_ctc "${OUT_100H}" $((PORT_BASE + 21))
fi

echo "DONE ${CONDITION}: ${OUT_10H}/eval_results.json and ${OUT_100H}/eval_results.json"
