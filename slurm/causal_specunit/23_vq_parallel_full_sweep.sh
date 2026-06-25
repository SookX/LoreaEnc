#!/bin/bash
# Full 9M-parameter parallel-VQ sweep at SqueezeFormer-XS scale.
#
# Submits 18 fine-tune + eval cells as a slurm array:
#   {iter-1, iter-2} parallel-VQ x {1h, 10h, 100h} x {seed 42, 43, 44}
#
# This is the publication-grade table for the 9M parallel-VQ recipe.
# After this completes you have mean +/- std across 3 seeds for every
# (iter, budget) cell, ready for direct comparison against HuBERT-Base /
# wav2vec2-Base (when you build the larger-variant comparison).
#
# All cells reuse the existing SSL pretrained checkpoints:
#   iter-1: outputs/causal_specunit/ssl_parallel_iter1_150k/checkpoint_step150000/checkpoint.pt
#   iter-2: outputs/causal_specunit/ssl_parallel_iter2_100k/checkpoint_step100000/checkpoint.pt
#
# Stages skipped (already done in 21_/22_ smokes):
#   A. VQ target generation
#   B. SSL pretrain
# Stages run per cell:
#   D. CTC fine-tune (auto-batch-size per budget)
#   E. Evaluate on test-clean / test-other
#
# Auto-skips cells whose eval_results.json already exists (the seed-42
# cells you already ran).
#
# Submit:
#   sbatch slurm/causal_specunit/23_vq_parallel_full_sweep.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=vq_par_sw
#SBATCH --array=0-17
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/vq_par_sw.%A_%a.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/vq_par_sw.%A_%a.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

# -------------------- Cell mapping --------------------
# Array index 0..17 -> (iter, budget, seed)
# Layout: ITER outer (2), BUDGET middle (3), SEED inner (3). So 2*3*3 = 18.
ITERS=(iter1 iter2)
BUDGETS=(librilight_1h librilight_10h train-clean-100)
SEEDS=(42 43 44)

IDX="${SLURM_ARRAY_TASK_ID:?must be run as a slurm array job}"
ITER_IDX=$((IDX / 9))
REMAINDER=$((IDX % 9))
BUDGET_IDX=$((REMAINDER / 3))
SEED_IDX=$((REMAINDER % 3))

ITER="${ITERS[${ITER_IDX}]}"
SUBSET="${BUDGETS[${BUDGET_IDX}]}"
SEED="${SEEDS[${SEED_IDX}]}"

# Per-budget batch size on 2 GPUs to match script 10's effective batches.
# 1h: 32 eff, 10h: 128 eff, 100h: 512 eff.
case "${SUBSET}" in
    librilight_1h)   FT_BATCH_SIZE=16 ;;
    librilight_10h)  FT_BATCH_SIZE=64 ;;
    train-clean-100) FT_BATCH_SIZE=256 ;;
    *) echo "Unknown SUBSET=${SUBSET}"; exit 1 ;;
esac

# Per-iter SSL checkpoint + output root (must match the layout the
# 21_/22_ smoke scripts already used).
if [ "${ITER}" = "iter1" ]; then
    SSL_CKPT="outputs/causal_specunit/ssl_parallel_iter1_150k/checkpoint_step150000/checkpoint.pt"
    OUTPUT_ROOT="outputs/causal_specunit/vq_smoke/parallel"
else
    SSL_CKPT="outputs/causal_specunit/ssl_parallel_iter2_100k/checkpoint_step100000/checkpoint.pt"
    OUTPUT_ROOT="outputs/causal_specunit/vq_iter2/parallel"
fi

FT_OUT_DIR="${OUTPUT_ROOT}/${SUBSET}/seed${SEED}"

# -------------------- Shared config --------------------
PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
SOURCE_TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
TOKENIZER_PATH="dataset/bpe128.model"
NPROC_PER_NODE="2"
FT_EPOCHS="${FT_EPOCHS:-150}"

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

log_phase "CELL ${IDX} | iter=${ITER} subset=${SUBSET} seed=${SEED} batch=${FT_BATCH_SIZE} (eff. $((FT_BATCH_SIZE * NPROC_PER_NODE)))"
echo "SSL checkpoint: ${SSL_CKPT}"
echo "Output dir:     ${FT_OUT_DIR}"

# -------------------- Sanity checks --------------------
[ -d "${VIRTUAL_ENV}" ]                          || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}" ]                            || { echo "Missing data root: ${DATA_ROOT}"; exit 1; }
[ -f "${SOURCE_TARGETS_DIR}/cmvn.pt" ]           || { echo "Missing CMVN"; exit 1; }
[ -f "${SSL_CKPT}" ]                             || { echo "Missing SSL checkpoint: ${SSL_CKPT}"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]                       || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }

# -------------------- Skip-if-done --------------------
if [ -f "${FT_OUT_DIR}/eval_results.json" ]; then
    log_phase "SKIP cell: ${FT_OUT_DIR}/eval_results.json already exists"
    exit 0
fi

PORT_BASE=$((39000 + (SLURM_JOB_ID % 2500) + IDX * 7))

# -------------------- Stage D: CTC fine-tune --------------------
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
        --variant xs \
        --epochs "${FT_EPOCHS}" \
        --batch-size "${FT_BATCH_SIZE}" \
        --grad-accum-steps 1 \
        --eval-batch-size 128 \
        --eval-split dev-other \
        --eval-every 1 \
        --workers 8 \
        --dataloader-timeout 120 \
        --lr 1e-3 \
        --encoder-lr 3e-4 \
        --head-lr 1e-3 \
        --warmup-epochs 10 \
        --peak-epochs 50 \
        --noam-decay-rate 0.5 \
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

# -------------------- Stage E: Evaluate --------------------
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
    --variant xs \
    --splits test-clean test-other \
    --batch-size 64 \
    --workers 4 \
    --output "${FT_OUT_DIR}/eval_results.json"

log_phase "DONE cell ${IDX}: iter=${ITER} subset=${SUBSET} seed=${SEED}"
echo "Result: ${FT_OUT_DIR}/eval_results.json"
