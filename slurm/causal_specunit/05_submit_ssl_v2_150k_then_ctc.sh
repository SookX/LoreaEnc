#!/bin/bash
# Single-allocation overnight run:
#   1) SSL v2 pretraining to 150k steps
#   2) CTC fine-tuning with LP-FT + InterCTC from the new checkpoint
#
# Run from the repo root on the cluster:
#   sbatch slurm/causal_specunit/05_submit_ssl_v2_150k_then_ctc.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_sslv2_150k_ctc
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_sslv2_150k_ctc.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_sslv2_150k_ctc.%j.err

set -euo pipefail

SLURM_ACCOUNT="${SLURM_JOB_ACCOUNT:-bg-eng-01}"
PROJECT_DIR="${PROJECT_DIR:-/valhalla/projects/${SLURM_ACCOUNT}/LoreaEnc}"
PRETRAIN_SCRIPT="${PRETRAIN_SCRIPT:-slurm/causal_specunit/02_pretrain_ssl_v2_100k_c8.sh}"
CTC_SCRIPT="${CTC_SCRIPT:-slurm/causal_specunit/03_train_ctc_150ep_fair_ssl_lpft_interctc.sh}"

MAX_STEPS="${MAX_STEPS:-150000}"
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-outputs/causal_specunit/pretrain_ssl_v2_150k_c8}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-${PRETRAIN_OUTPUT_DIR}/checkpoint_step150000/checkpoint.pt}"
CTC_OUTPUT_DIR="${CTC_OUTPUT_DIR:-outputs/causal_specunit/ctc_ssl_960h_v2_150k_lpft_interctc_elr6e4_ld085_w10_p90_d025_150ep_c8}"

if [ ! -d "${PROJECT_DIR}" ]; then
    echo "Missing project dir: ${PROJECT_DIR}"
    exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs

echo "Single-allocation SSL v2 150k -> CTC LP-FT + InterCTC starting at $(date)"
echo "Project dir: ${PROJECT_DIR}"
echo "Pretrain script: ${PRETRAIN_SCRIPT}"
echo "Pretrain output: ${PRETRAIN_OUTPUT_DIR}"
echo "Pretrain max steps: ${MAX_STEPS}"
echo "CTC script: ${CTC_SCRIPT}"
echo "CTC checkpoint: ${PRETRAIN_CHECKPOINT}"
echo "CTC output: ${CTC_OUTPUT_DIR}"

MAX_STEPS="${MAX_STEPS}" \
OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR}" \
bash "${PRETRAIN_SCRIPT}"

if [ ! -f "${PRETRAIN_CHECKPOINT}" ]; then
    echo "Expected pretrain checkpoint was not created: ${PRETRAIN_CHECKPOINT}"
    exit 1
fi

SSL_CHECKPOINT="${PRETRAIN_CHECKPOINT}" \
OUTPUT_DIR="${CTC_OUTPUT_DIR}" \
bash "${CTC_SCRIPT}"

echo "Single-allocation SSL v2 150k -> CTC LP-FT + InterCTC finished at $(date)"
echo "Pretrain checkpoint: ${PRETRAIN_CHECKPOINT}"
echo "CTC metrics: ${CTC_OUTPUT_DIR}/ctc_metrics.jsonl"
