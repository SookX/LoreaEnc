#!/bin/bash
# Dual-codebook ablation: parameterized SSL pretraining.
#
# Required env: CODEBOOK_MODE in {coarse, fine, both}
# Optional env: SSL_OUTPUT_DIR, MAX_STEPS (default 50000)
#
# Output: outputs/causal_specunit/abl_codebook/ssl_<mode>/checkpoint_step050000/

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_abl_ssl
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_abl_ssl.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_abl_ssl.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"

: "${CODEBOOK_MODE:?Required: CODEBOOK_MODE in {coarse,fine,both}}"
case "${CODEBOOK_MODE}" in
    coarse|fine|both) ;;
    *) echo "Invalid CODEBOOK_MODE=${CODEBOOK_MODE} (must be coarse|fine|both)"; exit 1 ;;
esac

MAX_STEPS="${MAX_STEPS:-50000}"
SSL_OUTPUT_DIR="${SSL_OUTPUT_DIR:-outputs/causal_specunit/abl_codebook/ssl_${CODEBOOK_MODE}}"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
cd "${PROJECT_DIR}"
mkdir -p logs "${SSL_OUTPUT_DIR}"

for f in "${TARGETS_DIR}/targets.pt" "${TARGETS_DIR}/cmvn.pt" "${TARGETS_DIR}/metadata.json"; do
    [ -f "${f}" ] || { echo "Missing: ${f}"; exit 1; }
done
if [ ! -f "${TARGETS_DIR}/target_index.json" ]; then
    python -m CausalSpecUnit.shard_targets --targets-dir "${TARGETS_DIR}" --num-shards 128
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((15000 + SLURM_JOB_ID % 20000))}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

NUM_PROCESSES=4
BATCH_SIZE=64
GRAD_ACCUM_STEPS=1
WORKERS=8

echo "Job ${SLURM_JOB_ID} | CODEBOOK_MODE=${CODEBOOK_MODE} | MAX_STEPS=${MAX_STEPS}"
echo "Output: ${SSL_OUTPUT_DIR}"
echo "Effective batch: $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
echo "Date: $(date)"

# Same v2 recipe as the main run (LayerDrop, aux@8, SpecAug, mask 0.50).
# Only --codebook-mode varies across the three ablation legs.
torchrun \
    --nproc_per_node="${NUM_PROCESSES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m CausalSpecUnit.pretrain_ssl \
    --data-root "${DATA_ROOT}" \
    --targets-dir "${TARGETS_DIR}" \
    --output-dir "${SSL_OUTPUT_DIR}" \
    --variant xs \
    --epochs 1000 \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --codebook-mode "${CODEBOOK_MODE}" \
    --mask-prob 0.50 \
    --mask-length 10 \
    --chunk-size 8 \
    --chunk-stride 4 \
    --layer-drop-p 0.05 \
    --aux-layers 8 \
    --aux-weight 0.5 \
    --specaug \
    --specaug-time-mask-param 30 \
    --specaug-freq-mask-param 20 \
    --specaug-time-masks 2 \
    --specaug-freq-masks 2 \
    --lr 1e-3 \
    --warmup-epochs 10 \
    --peak-epochs 10 \
    --noam-decay-rate 1.0 \
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 200.0 \
    --workers "${WORKERS}" \
    --dataloader-timeout 300 \
    --prefetch-factor 4 \
    --log-every 100 \
    --save-every 5 \
    --progress off

echo "Done at $(date)"
echo "Final checkpoint: ${SSL_OUTPUT_DIR}/checkpoint_step$(printf '%06d' ${MAX_STEPS})"
