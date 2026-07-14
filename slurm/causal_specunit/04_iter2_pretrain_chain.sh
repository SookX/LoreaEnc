#!/bin/bash
# Iter-2 SSL pretraining chain: generate iter-2 targets from the v2 150k
# iter-1 checkpoint, then SSL pretrain iter-2 with the simple recipe.
#
# Phases (sequential, same allocation):
#   1. Iter-2 target generation (uses 1 of the 2 allocated GPUs, ~6h)
#   2. Iter-2 SSL pretrain      (uses both 2 GPUs,          ~18-22h)
#
# Idempotent: each phase checks for its output and skips if present.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_iter2_chain
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_iter2_chain.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_iter2_chain.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"

# Source iter-1 checkpoint (the v2 150k that's already on disk)
SOURCE_SSL_CHECKPOINT="${SOURCE_SSL_CHECKPOINT:-outputs/causal_specunit/pretrain_ssl_v2_150k_c8/checkpoint_step150000/checkpoint.pt}"
SOURCE_TARGETS_DIR="${SOURCE_TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"

# Output directories for the two phases
ITER2_TARGETS_DIR="${ITER2_TARGETS_DIR:-outputs/causal_specunit/targets_iter2_from_v2_c8}"
ITER2_SSL_OUTPUT_DIR="${ITER2_SSL_OUTPUT_DIR:-outputs/causal_specunit/pretrain_ssl_iter2_from_v2_c8}"

# Iter-2 SSL knobs (simple recipe, identical to old iter-1)
MAX_STEPS="${MAX_STEPS:-100000}"
LR="${LR:-1e-3}"
MASK_PROB="${MASK_PROB:-0.30}"
MASK_LENGTH="${MASK_LENGTH:-10}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
PEAK_EPOCHS="${PEAK_EPOCHS:-20}"
NOAM_DECAY_RATE="${NOAM_DECAY_RATE:-1.0}"

if [ ! -d "${VIRTUAL_ENV}" ]; then
    echo "Missing venv: ${VIRTUAL_ENV}"; exit 1
fi

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "${PROJECT_DIR}"
mkdir -p logs "${ITER2_TARGETS_DIR}" "${ITER2_SSL_OUTPUT_DIR}"

# Sanity checks
[ -d "${DATA_ROOT}" ]                       || { echo "Missing data: ${DATA_ROOT}"; exit 1; }
[ -f "${SOURCE_TARGETS_DIR}/cmvn.pt" ]      || { echo "Missing source CMVN"; exit 1; }
if [ ! -f "${SOURCE_SSL_CHECKPOINT}" ] && [ ! -f "${SOURCE_SSL_CHECKPOINT}/checkpoint.pt" ]; then
    echo "Missing source SSL (iter-1) checkpoint: ${SOURCE_SSL_CHECKPOINT}"
    exit 1
fi

PORT_BASE=$((19000 + SLURM_JOB_ID % 10000))
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

log_phase() {
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "============================================================"
}

# ---------- Phase 1: generate iter-2 targets from v2 150k iter-1 ----------
ITER2_TARGETS_FILE="${ITER2_TARGETS_DIR}/targets.pt"
if [ -f "${ITER2_TARGETS_FILE}" ] && [ -f "${ITER2_TARGETS_DIR}/metadata.json" ]; then
    log_phase "SKIP iter-2 target generation — targets already exist at ${ITER2_TARGETS_DIR}"
else
    log_phase "PHASE 1: iter-2 target generation from ${SOURCE_SSL_CHECKPOINT}"
    python -m CausalSpecUnit.generate_iter2_targets \
        --data-root "${DATA_ROOT}" \
        --splits train-clean-100 train-clean-360 train-other-500 \
        --cmvn-path "${SOURCE_TARGETS_DIR}/cmvn.pt" \
        --ssl-checkpoint "${SOURCE_SSL_CHECKPOINT}" \
        --output-dir "${ITER2_TARGETS_DIR}" \
        --variant xs \
        --chunk-size 8 \
        --chunk-stride 4 \
        --pca-dim 64 \
        --k-coarse 100 \
        --k-fine 500 \
        --max-fit-frames 1000000 \
        --fit-frames-per-batch 8192 \
        --batch-size 32 \
        --workers 8 \
        --dataloader-timeout 180 \
        --target-shards 128 \
        --seed 42
fi

# Shard targets if not already sharded (the SSL pretrain reads sharded targets)
if [ ! -f "${ITER2_TARGETS_DIR}/target_index.json" ]; then
    log_phase "Sharding iter-2 targets"
    python -m CausalSpecUnit.shard_targets --targets-dir "${ITER2_TARGETS_DIR}" --num-shards 128
fi

# ---------- Phase 2: iter-2 SSL pretrain (2 GPUs) ----------
ITER2_SSL_CKPT="${ITER2_SSL_OUTPUT_DIR}/checkpoint_step$(printf '%06d' ${MAX_STEPS})/checkpoint.pt"
if [ -f "${ITER2_SSL_CKPT}" ]; then
    log_phase "SKIP iter-2 SSL pretrain — checkpoint exists at ${ITER2_SSL_CKPT}"
else
    log_phase "PHASE 2: iter-2 SSL pretrain on ${ITER2_TARGETS_DIR} (warm-start from ${SOURCE_SSL_CHECKPOINT})"
    export MASTER_PORT=$((PORT_BASE + 100))
    torchrun \
        --nproc_per_node=2 \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m CausalSpecUnit.pretrain_ssl \
        --data-root "${DATA_ROOT}" \
        --targets-dir "${ITER2_TARGETS_DIR}" \
        --output-dir "${ITER2_SSL_OUTPUT_DIR}" \
        --variant xs \
        --epochs 1000 \
        --max-steps "${MAX_STEPS}" \
        --batch-size 128 \
        --grad-accum-steps 1 \
        --init-encoder-checkpoint "${SOURCE_SSL_CHECKPOINT}" \
        --mask-prob "${MASK_PROB}" \
        --mask-length "${MASK_LENGTH}" \
        --chunk-size 8 \
        --chunk-stride 4 \
        --lr "${LR}" \
        --warmup-epochs "${WARMUP_EPOCHS}" \
        --peak-epochs "${PEAK_EPOCHS}" \
        --noam-decay-rate "${NOAM_DECAY_RATE}" \
        --max-grad-norm 1.0 \
        --max-safe-grad-norm 200.0 \
        --workers 12 \
        --dataloader-timeout 300 \
        --prefetch-factor 4 \
        --log-every 100 \
        --save-every 10 \
        --trace-startup \
        --progress off
fi

log_phase "DONE iter-2 pretrain chain (job ${SLURM_JOB_ID})"
echo "Iter-2 targets: ${ITER2_TARGETS_DIR}"
echo "Iter-2 SSL ckpt: ${ITER2_SSL_CKPT}"
