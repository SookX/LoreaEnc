#!/bin/bash
# Standalone MelHuBERT-Transformer (mh9m) fine-tune. Bypasses train_ctc.py
# entirely; uses CausalSpecUnit/finetune_mh9m.py which is a self-contained
# CTC fine-tune that only depends on the encoder class and the dataset.
#
# Hyperparameters mirror 10_benchmark_1h_10h_100h_3seeds.sh:
#   epochs=150, warmup=10, peak=50, noam_decay=0.5
#   encoder-lr=3e-4 head-lr=1e-3 weight-decay=5e-4 max-grad-norm=1.0
#   specaug 30/20/2/2 (time/freq mask params, mask counts)
#   per-split batch: 8 (1h), 32 (10h), 128 (100h) — same as script 10
#
# Submit (single cell, useful for fast iteration):
#   FORCE_TRAIN=1 SUBSET=librilight_10h SEED=42 \
#     sbatch slurm/causal_specunit/16_melhubert_finetune_standalone.sh
#
# Submit (the full 3 seeds x 3 subsets sweep mirroring script 10):
#   sbatch --array=0-8 slurm/causal_specunit/16_melhubert_finetune_standalone.sh
#
# Optional knobs:
#   WORKERS=0 sbatch ...
#   EPOCHS=10 sbatch ...     # quick sanity run

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=mh9m_ft
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/mh9m_ft.%A_%a.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/mh9m_ft.%A_%a.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-dataset/bpe128.model}"
SSL_CKPT="${SSL_CKPT:-outputs/causal_specunit/melhubert_transformer_mh9m/ssl_fine_150000/checkpoint_step150000/checkpoint.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/causal_specunit/mh9m_ft}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-150}"

# Resolve (SUBSET, SEED) from either env vars (single submission) or
# SLURM_ARRAY_TASK_ID (3 subsets x 3 seeds = 9 cells).
SUBSETS=(librilight_1h librilight_10h train-clean-100)
SEEDS=(42 43 44)

if [ -n "${SUBSET:-}" ] && [ -n "${SEED:-}" ]; then
    : # use what was set
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    IDX="${SLURM_ARRAY_TASK_ID}"
    SUBSET="${SUBSETS[$((IDX / 3))]}"
    SEED="${SEEDS[$((IDX % 3))]}"
else
    SUBSET="${SUBSET:-librilight_10h}"
    SEED="${SEED:-42}"
fi

# Per-split batch size, matching 10_benchmark_1h_10h_100h_3seeds.sh.
case "${SUBSET}" in
    librilight_1h)   BATCH_SIZE="${BATCH_SIZE:-8}" ;;
    librilight_10h)  BATCH_SIZE="${BATCH_SIZE:-32}" ;;
    train-clean-100) BATCH_SIZE="${BATCH_SIZE:-128}" ;;
    *) echo "Unsupported SUBSET=${SUBSET}"; exit 1 ;;
esac

OUT_DIR="${OUTPUT_ROOT}/${SUBSET}/seed${SEED}"

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
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
mkdir -p logs "${OUT_DIR}"

[ -d "${VIRTUAL_ENV}" ]            || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}/${SUBSET}" ]    || { echo "Missing train split: ${DATA_ROOT}/${SUBSET}"; exit 1; }
[ -d "${DATA_ROOT}/dev-other" ]    || { echo "Missing dev split: ${DATA_ROOT}/dev-other"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]    || { echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]         || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }
[ -f "${SSL_CKPT}" ]               || { echo "Missing SSL checkpoint: ${SSL_CKPT}"; exit 1; }

if [ "${FORCE_TRAIN:-0}" = "1" ] || [ "${CLEAN_FIRST:-0}" = "1" ]; then
    find "${OUT_DIR}" -mindepth 1 ! -name ".nfs*" -delete 2>/dev/null || true
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
PORT_BASE=$((33000 + (SLURM_JOB_ID % 2500)))

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] mh9m FT standalone | subset=${SUBSET} seed=${SEED} batch=${BATCH_SIZE} workers=${WORKERS}"
echo "  out=${OUT_DIR}"
echo "===================================================="

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${PORT_BASE}" \
    -m CausalSpecUnit.finetune_mh9m \
    --data-root "${DATA_ROOT}" \
    --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --ssl-checkpoint "${SSL_CKPT}" \
    --output-dir "${OUT_DIR}" \
    --train-split "${SUBSET}" \
    --dev-split dev-other \
    --variant mh9m \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size 64 \
    --workers "${WORKERS}" \
    --encoder-lr 3e-4 \
    --head-lr 1e-3 \
    --weight-decay 5e-4 \
    --warmup-epochs 10 \
    --peak-epochs 50 \
    --noam-decay-rate 0.5 \
    --max-grad-norm 1.0 \
    --specaug-time-mask-param 30 \
    --specaug-freq-mask-param 20 \
    --specaug-time-masks 2 \
    --specaug-freq-masks 2 \
    --seed "${SEED}"

echo "DONE mh9m FT subset=${SUBSET} seed=${SEED}"
