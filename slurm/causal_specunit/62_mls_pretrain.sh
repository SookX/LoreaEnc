#!/bin/bash
# Iter-1 spectrogram SSL pretraining for one MLS language, on the targets from
# 61_mls_targets.sh. Same recipe as the English 02_pretrain_ssl run (dual
# codebook K=100/500, mask 0.30/10, chunk 8/4, lr 1e-3) but only 50k steps by
# default -- enough to establish the scratch < iter-1 ordering in a new language
# on the calendar. Auto-chains across the 4h wall cap.
#
# Prereq: MLS_LANG=<lang> sbatch slurm/causal_specunit/61_mls_targets.sh  (targets first)
# Submit:
#   MLS_LANG=polish sbatch slurm/causal_specunit/62_mls_pretrain.sh
#   MLS_LANG=polish MAX_STEPS=150000 sbatch ...   # full iter-1 budget if time allows

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=mls_ssl
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/mls_ssl.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/mls_ssl.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"

MLS_LANG="${MLS_LANG:-polish}"
MLS_LANG_ROOT="${MLS_LANG_ROOT:-dataset/mls/mls_${MLS_LANG}}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/mls_targets_${MLS_LANG}_c8}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/mls_ssl_${MLS_LANG}_iter1_50k}"
MAX_STEPS="${MAX_STEPS:-50000}"
CODEBOOK_MODE="${CODEBOOK_MODE:-both}"
PEAK_EPOCHS="${PEAK_EPOCHS:-40}"

[ -d "${VIRTUAL_ENV}" ]                 || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${MLS_LANG_ROOT}/train/audio" ]   || { echo "Missing MLS audio: ${MLS_LANG_ROOT}/train/audio"; exit 1; }
[ -f "${TARGETS_DIR}/targets.pt" ]      || { echo "Missing targets: ${TARGETS_DIR}/targets.pt (run 61_mls_targets.sh first)"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]         || { echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"; exit 1; }
[ -f "${TARGETS_DIR}/metadata.json" ]   || { echo "Missing metadata: ${TARGETS_DIR}/metadata.json"; exit 1; }

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -f "${TARGETS_DIR}/target_index.json" ]; then
    echo "Sharding targets (${TARGETS_DIR})..."
    python -m CausalSpecUnit.shard_targets --targets-dir "${TARGETS_DIR}" --num-shards 128
fi

export TARGETS_DIR
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((13500 + SLURM_JOB_ID % 20000))}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

NUM_PROCESSES=2
WORKERS=12
DATALOADER_TIMEOUT=300

echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] MLS SSL iter-1: ${MLS_LANG}"
echo "  mls_lang_root=${MLS_LANG_ROOT}  targets=${TARGETS_DIR}"
echo "  output=${OUTPUT_DIR}  max_steps=${MAX_STEPS}  codebook=${CODEBOOK_MODE}"
echo "===================================================="

# 4h wall-cap auto-chaining: resolve latest checkpoint, exit if MAX_STEPS reached,
# else set RESUME_CKPT and queue a successor. Propagates env via --export=ALL.
SELF_SCRIPT="slurm/causal_specunit/62_mls_pretrain.sh"
source slurm/causal_specunit/_autochain.sh

RESUME_CKPT="${RESUME_CKPT:-}"
if [ -n "${RESUME_CKPT}" ]; then
    [ -f "${RESUME_CKPT}/checkpoint.pt" ] || { echo "RESUME_CKPT set but checkpoint.pt not found: ${RESUME_CKPT}"; exit 1; }
    echo "Resuming from checkpoint: ${RESUME_CKPT}"
fi

torchrun \
    --nproc_per_node="${NUM_PROCESSES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m CausalSpecUnit.pretrain_ssl \
    --mls-lang-root "${MLS_LANG_ROOT}" \
    --targets-dir "${TARGETS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --variant xs \
    --codebook-mode "${CODEBOOK_MODE}" \
    --epochs 1000 \
    --max-steps "${MAX_STEPS}" \
    --batch-size 128 \
    --grad-accum-steps 1 \
    --mask-prob 0.30 \
    --mask-length 10 \
    --chunk-size 8 \
    --chunk-stride 4 \
    --lr 1e-3 \
    --warmup-epochs 20 \
    --peak-epochs "${PEAK_EPOCHS}" \
    --noam-decay-rate 1.0 \
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 200.0 \
    --workers "${WORKERS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --prefetch-factor 4 \
    --log-every 10 \
    --save-every 1 \
    --trace-startup \
    --progress on \
    $( [ -n "${RESUME_CKPT}" ] && echo "--resume ${RESUME_CKPT}" || true )
