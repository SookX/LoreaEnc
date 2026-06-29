#!/bin/bash
# SqueezeFormer-M95 fine-tune + eval on Libri-Light 1h (seed 42) using
# the iter-1 400k SSL checkpoint. Single-cell validation: are we getting
# something competitive with the 95M-scale baselines?
#
# Reference numbers to beat:
#   - parallel-VQ iter-2 at 9M, 1h: 46.9 / 59.1 test-clean/test-other (our paper)
#   - wav2vec2-Base 1h with LM: 5.5 / 11.3 (their paper)
#   - wav2vec2-Base 1h no LM: ~24 / ~30 (estimated)
#
# Submit (after iter-1 SSL checkpoint exists):
#   sbatch slurm/causal_specunit/41_finetune_m95_1h_seed42.sh
#
# Optional knobs:
#   SSL_CKPT=...  override the SSL checkpoint path
#   SEED=43       different seed
#   FT_BATCH_SIZE=8   override per-GPU batch

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=m95_ft1h
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/m95_ft1h.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/m95_ft1h.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
SOURCE_TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
TOKENIZER_PATH="dataset/bpe128.model"
NPROC_PER_NODE="2"

SSL_CKPT="${SSL_CKPT:-outputs/causal_specunit/ssl_m95_iter1_400k/checkpoint_step400000/checkpoint.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/causal_specunit/m95_smoke/iter1}"
SUBSET="${SUBSET:-librilight_1h}"
SEED="${SEED:-42}"
FT_BATCH_SIZE="${FT_BATCH_SIZE:-16}"   # 2 GPUs x 16 = 32 effective, matches 9M iter-2 1h recipe (script 10)
FT_EPOCHS="${FT_EPOCHS:-150}"

# Hyperparameters that diagnostics need to override.
# Defaults match the 9M recipe; tests can lower the LRs or freeze the encoder.
ENCODER_LR="${ENCODER_LR:-1e-4}"
HEAD_LR="${HEAD_LR:-1e-3}"
BASE_LR="${BASE_LR:-1e-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
PEAK_EPOCHS="${PEAK_EPOCHS:-50}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-0}"
REWARMUP_EPOCHS="${REWARMUP_EPOCHS:-0}"
ALLOW_INTERMEDIATE_CKPT="${ALLOW_INTERMEDIATE_CKPT:-0}"
TAG="${TAG:-}"

# Tag the output dir so diagnostic runs (frozen, lowlr, etc.) don't collide.
if [ -n "${TAG}" ]; then
    FT_OUT_DIR="${OUTPUT_ROOT}/${SUBSET}/seed${SEED}_${TAG}"
else
    FT_OUT_DIR="${OUTPUT_ROOT}/${SUBSET}/seed${SEED}"
fi

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
mkdir -p logs "${FT_OUT_DIR}"

log_phase() {
    echo ""
    echo "===================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "===================================================="
}

log_phase "M95 1h fine-tune | subset=${SUBSET} seed=${SEED} batch=${FT_BATCH_SIZE}"
echo "SSL checkpoint: ${SSL_CKPT}"
echo "Output dir:     ${FT_OUT_DIR}"

[ -d "${VIRTUAL_ENV}" ]                 || { echo "Missing venv"; exit 1; }
[ -d "${DATA_ROOT}" ]                   || { echo "Missing data root"; exit 1; }
[ -f "${SOURCE_TARGETS_DIR}/cmvn.pt" ]  || { echo "Missing CMVN"; exit 1; }
# If the default 400k checkpoint doesn't exist yet, optionally fall back to the
# most-recent existing checkpoint in the same dir. This is intentionally gated:
# otherwise a nominal "400k" validation can silently use a 150k undertrained
# encoder and produce misleading fine-tune results.
if [ ! -f "${SSL_CKPT}" ]; then
    SSL_DIR=$(dirname "$(dirname "${SSL_CKPT}")")
    AUTO_CKPT=$(ls -t "${SSL_DIR}"/checkpoint_step*/checkpoint.pt 2>/dev/null | head -1)
    if [ "${ALLOW_INTERMEDIATE_CKPT}" = "1" ] && [ -n "${AUTO_CKPT}" ]; then
        echo "SSL_CKPT default not found; falling back to latest: ${AUTO_CKPT}"
        SSL_CKPT="${AUTO_CKPT}"
    elif [ -n "${AUTO_CKPT}" ]; then
        echo "Missing requested SSL checkpoint: ${SSL_CKPT}"
        echo "Latest intermediate exists: ${AUTO_CKPT}"
        echo "Set ALLOW_INTERMEDIATE_CKPT=1 or pass SSL_CKPT=${AUTO_CKPT} to use it intentionally."
        exit 1
    fi
fi
[ -f "${SSL_CKPT}" ]                    || { echo "Missing SSL checkpoint: ${SSL_CKPT}"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]              || { echo "Missing tokenizer"; exit 1; }

if [ -f "${FT_OUT_DIR}/eval_results.json" ]; then
    log_phase "SKIP cell: ${FT_OUT_DIR}/eval_results.json already exists"
    exit 0
fi

PORT_BASE=$((45000 + (SLURM_JOB_ID % 2500)))

if [ ! -f "${FT_OUT_DIR}/checkpoint_best/checkpoint.pt" ]; then
    log_phase "Stage D: CTC fine-tune"
    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="$((PORT_BASE + 1))" \
        -m CausalSpecUnit.train_ctc \
        --data-root "${DATA_ROOT}" \
        --cmvn-path "${SOURCE_TARGETS_DIR}/cmvn.pt" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --train-splits "${SUBSET}" \
        --ssl-checkpoint "${SSL_CKPT}" \
        --output-dir "${FT_OUT_DIR}" \
        --variant m95 \
        --epochs "${FT_EPOCHS}" \
        --batch-size "${FT_BATCH_SIZE}" \
        --grad-accum-steps 1 \
        --eval-batch-size 64 \
        --eval-split dev-other \
        --eval-every 1 \
        --workers 8 \
        --dataloader-timeout 120 \
        --lr "${BASE_LR}" \
        --encoder-lr "${ENCODER_LR}" \
        --head-lr "${HEAD_LR}" \
        --warmup-epochs "${WARMUP_EPOCHS}" \
        --peak-epochs "${PEAK_EPOCHS}" \
        --noam-decay-rate 0.5 \
        --freeze-encoder-epochs "${FREEZE_EPOCHS}" \
        --encoder-rewarmup-epochs "${REWARMUP_EPOCHS}" \
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
    log_phase "Stage D: SKIP (best checkpoint already exists)"
fi

log_phase "Stage E: evaluate test-clean + test-other"
torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="$((PORT_BASE + 2))" \
    -m CausalSpecUnit.evaluate_ctc \
    --checkpoint "${FT_OUT_DIR}/checkpoint_best/checkpoint.pt" \
    --data-root "${DATA_ROOT}" \
    --cmvn-path "${SOURCE_TARGETS_DIR}/cmvn.pt" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --variant m95 \
    --splits test-clean test-other \
    --batch-size 32 \
    --workers 4 \
    --output "${FT_OUT_DIR}/eval_results.json"

log_phase "DONE m95 1h iter=1 seed=${SEED}"
echo "Result: ${FT_OUT_DIR}/eval_results.json"
echo ""
echo "Compare against:"
echo "  9M parallel-VQ iter-2, 1h:  46.9 / 59.1 test-clean/test-other (no LM)"
echo "  wav2vec2-Base 1h, with LM:   5.5 / 11.3"
echo "  wav2vec2-Base 1h, no LM:    ~24 / ~30 (estimated)"
