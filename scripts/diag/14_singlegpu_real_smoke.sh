#!/bin/bash
# Single-GPU training smoke with the REAL Libri-Light 10h dataloader.
# No DDP, no NCCL. If this works, the issue in the failing 4-GPU run is
# in DDP / NCCL / dataloader-worker concurrency, not the model or the
# data path itself.
#
# Behavior on this script:
#   - 1 GPU
#   - workers=0 (synchronous dataloader) by default
#   - 3 epochs to exercise the eval-every-epoch boundary
#   - FORCE_TRAIN=1 wipes the output dir
#
# Submit:
#   sbatch scripts/diag/14_singlegpu_real_smoke.sh
#
# Overrides:
#   WORKERS=4 sbatch ...                  # try 4-worker dataloader
#   SUBSET=librilight_1h sbatch ...
#   EPOCHS=1 sbatch ...

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=diag_1gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/diag_1gpu.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/diag_1gpu.%j.err

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

SUBSET="${SUBSET:-librilight_10h}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-0}"
DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-300}"

OUT_DIR="${OUT_DIR:-outputs/causal_specunit/diag/diag_1gpu_${SUBSET}_w${WORKERS}}"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

cd "${PROJECT_DIR}"
mkdir -p logs "${OUT_DIR}"

# Wipe stale state, including .nfs* placeholders.
find "${OUT_DIR}" -mindepth 1 ! -name ".nfs*" -delete 2>/dev/null || true

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DIAG single-GPU smoke"
echo "  subset=${SUBSET} epochs=${EPOCHS} batch=${BATCH_SIZE} workers=${WORKERS}"
echo "  out=${OUT_DIR}"
echo "===================================================="

python -m CausalSpecUnit.train_ctc \
    --data-root "${DATA_ROOT}" \
    --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --train-splits "${SUBSET}" \
    --ssl-checkpoint "${SSL_CKPT}" \
    --output-dir "${OUT_DIR}" \
    --variant mh9m \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps 1 \
    --eval-batch-size 64 \
    --eval-split dev-other \
    --eval-every 1 \
    --workers "${WORKERS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --lr 1e-3 \
    --encoder-lr 3e-4 \
    --head-lr 1e-3 \
    --warmup-epochs 1 \
    --peak-epochs 1 \
    --noam-decay-rate 0.5 \
    --max-grad-norm 1.0 \
    --specaug \
    --specaug-time-mask-param 30 \
    --specaug-freq-mask-param 20 \
    --specaug-time-masks 2 \
    --specaug-freq-masks 2 \
    --specaug-disable-last-epochs 0 \
    --seed 42 \
    --progress off \
    --log-every 0 \
    --save-every 1

echo "DIAG single-GPU smoke DONE"
