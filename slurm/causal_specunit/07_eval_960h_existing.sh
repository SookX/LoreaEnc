#!/bin/bash
# Evaluate the already-trained 960h iter-1 and iter-2 fine-tunes on
# test-clean and test-other, and write the eval_results.json files to
# the same test_table/ layout as 06_test_table.sh uses, so the
# aggregator picks them up automatically.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_eval_960h
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_eval_960h.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_eval_960h.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
TOKENIZER_PATH="dataset/bpe128.model"

# Source checkpoints (already-trained 960h fine-tunes)
ITER1_CKPT="outputs/causal_specunit/ctc_ssl_v2_150k_simple_960h/checkpoint_best/checkpoint.pt"
ITER2_CKPT="outputs/causal_specunit/ctc_ssl_iter2_simple_960h/checkpoint_best/checkpoint.pt"

# Where to write eval_results.json — same layout as 06_test_table.sh
ITER1_OUT="outputs/causal_specunit/test_table/iter1_960h"
ITER2_OUT="outputs/causal_specunit/test_table/iter2_960h"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

cd "${PROJECT_DIR}"
mkdir -p logs "${ITER1_OUT}" "${ITER2_OUT}"

[ -f "${ITER1_CKPT}" ]      || { echo "Missing iter-1 checkpoint: ${ITER1_CKPT}"; exit 1; }
[ -f "${ITER2_CKPT}" ]      || { echo "Missing iter-2 checkpoint: ${ITER2_CKPT}"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ] || { echo "Missing CMVN"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]   || { echo "Missing tokenizer"; exit 1; }

PORT_BASE=$((25000 + SLURM_JOB_ID % 10000))
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

run_eval() {
    local ckpt="$1"; local out_dir="$2"; local label="$3"; local port="$4"
    local results="${out_dir}/eval_results.json"
    if [ -f "${results}" ]; then
        echo "SKIP ${label}: ${results} already exists"
        return 0
    fi
    echo ""
    echo "===================================================="
    echo "[$(date '+%H:%M:%S')] EVAL ${label}"
    echo "  checkpoint: ${ckpt}"
    echo "  output:     ${results}"
    echo "===================================================="
    export MASTER_PORT="${port}"
    torchrun \
        --nproc_per_node=1 \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m CausalSpecUnit.evaluate_ctc \
        --checkpoint "${ckpt}" \
        --data-root "${DATA_ROOT}" \
        --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --variant xs \
        --splits test-clean test-other \
        --batch-size 64 \
        --workers 4 \
        --output "${results}"
}

run_eval "${ITER1_CKPT}" "${ITER1_OUT}" "iter1_960h" $((PORT_BASE + 1))
run_eval "${ITER2_CKPT}" "${ITER2_OUT}" "iter2_960h" $((PORT_BASE + 2))

echo ""
echo "DONE — both eval_results.json written:"
echo "  ${ITER1_OUT}/eval_results.json"
echo "  ${ITER2_OUT}/eval_results.json"
