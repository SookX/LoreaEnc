#!/bin/bash
# Dual-codebook ablation: evaluate checkpoint_best on test-clean + test-other.
#
# Required env: FT_OUTPUT_DIR (the fine-tune dir; checkpoint_best/ lives inside)

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_abl_eval
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_abl_eval.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_abl_eval.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
TOKENIZER_PATH="dataset/bpe128.model"

: "${FT_OUTPUT_DIR:?Required: FT_OUTPUT_DIR}"

CKPT_PATH="${FT_OUTPUT_DIR}/checkpoint_best/checkpoint.pt"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
cd "${PROJECT_DIR}"
mkdir -p logs

[ -f "${CKPT_PATH}" ] || { echo "Missing checkpoint: ${CKPT_PATH}"; exit 1; }

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((30000 + SLURM_JOB_ID % 20000))}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

NUM_PROCESSES=2

echo "Job ${SLURM_JOB_ID} | eval checkpoint=${CKPT_PATH}"

torchrun \
    --nproc_per_node="${NUM_PROCESSES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m CausalSpecUnit.evaluate_ctc \
    --checkpoint "${CKPT_PATH}" \
    --data-root "${DATA_ROOT}" \
    --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --variant xs \
    --splits test-clean test-other \
    --batch-size 64 \
    --workers 4 \
    --output "${FT_OUTPUT_DIR}/eval_results.json"

echo "Done at $(date)"
echo "Results: ${FT_OUTPUT_DIR}/eval_results.json"
