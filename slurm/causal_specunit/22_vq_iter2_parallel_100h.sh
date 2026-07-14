#!/bin/bash
# VQ iter-2 test on 100h ONLY (parallel-VQ).
#
# Self-contained pipeline:
#   A. Generate iter-2 VQ targets (extract hidden states from iter-1 parallel-VQ
#      encoder, PCA + train new parallel-VQ on those, then quantize every utterance).
#   B. Shard targets.
#   C. SSL pretrain 100k more steps initialised from the iter-1 parallel-VQ encoder.
#   D. Fine-tune on LibriSpeech train-clean-100, seed 42, batch 256 (eff. 512 on 2 GPUs).
#   E. Evaluate test-clean / test-other.
#
# Decision rule (per RESUBMISSION_PLAN.md):
#   Iter-2 parallel-VQ at 100h must beat iter-1 parallel-VQ at 100h
#   (16.22 / 35.36) by >=0.5 WER points test-other to justify full iter-2 sweep.
#
# Prerequisite:
#   outputs/causal_specunit/ssl_parallel_iter1_150k/checkpoint_step150000/checkpoint.pt
#   This is the iter-1 parallel-VQ SSL checkpoint produced by 21_vq_smoke_parallel.sh.
#
# Submit:
#   sbatch slurm/causal_specunit/22_vq_iter2_parallel_100h.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=vq_par_i2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/vq_par_i2.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/vq_par_i2.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

# -------------------- Quantizer-specific config --------------------
QUANTIZER_TYPE="parallel"
ITER1_SSL_CKPT="outputs/causal_specunit/ssl_parallel_iter1_150k/checkpoint_step150000/checkpoint.pt"
ITER2_TARGETS_DIR="outputs/causal_specunit/targets_iter2_vq_parallel"
ITER2_SSL_DIR="outputs/causal_specunit/ssl_parallel_iter2_100k"
FT_OUTPUT_ROOT="outputs/causal_specunit/vq_iter2/parallel"
K_COARSE="100"
K_FINE="500"
K1="100"
K2="500"

# -------------------- Shared config --------------------
PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
SOURCE_TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
TOKENIZER_PATH="dataset/bpe128.model"
NPROC_PER_NODE="2"

ITER2_MAX_STEPS="${ITER2_MAX_STEPS:-100000}"
ITER2_WARMUP_EPOCHS="${ITER2_WARMUP_EPOCHS:-20}"
ITER2_PEAK_EPOCHS="${ITER2_PEAK_EPOCHS:-20}"
SUBSET="${SUBSET:-train-clean-100}"
SEED="${SEED:-42}"
FT_BATCH_SIZE="${FT_BATCH_SIZE:-256}"   # 2 GPUs * 256 = 512 effective, matches script 10's 100h cell
FT_EPOCHS="${FT_EPOCHS:-150}"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONFAULTHANDLER_TIMEOUT=300
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
mkdir -p logs "${ITER2_TARGETS_DIR}" "${ITER2_SSL_DIR}" "${FT_OUTPUT_ROOT}"

# -------------------- Sanity checks --------------------
[ -d "${VIRTUAL_ENV}" ]                          || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}" ]                            || { echo "Missing data root: ${DATA_ROOT}"; exit 1; }
[ -f "${SOURCE_TARGETS_DIR}/cmvn.pt" ]           || { echo "Missing CMVN at ${SOURCE_TARGETS_DIR}/cmvn.pt"; exit 1; }
[ -f "${ITER1_SSL_CKPT}" ]                       || { echo "Missing iter-1 SSL checkpoint: ${ITER1_SSL_CKPT}. Run 21_vq_smoke_parallel.sh first."; exit 1; }
[ -f "${TOKENIZER_PATH}" ]                       || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }

log_phase() {
    echo ""
    echo "===================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "===================================================="
}

PORT_BASE=$((38000 + (SLURM_JOB_ID % 2500)))

# ==========================================================
# Stage A: Extract hidden states with iter-1 parallel-VQ encoder,
#          train new parallel-VQ on them, quantize every utterance.
# ==========================================================
if [ ! -f "${ITER2_TARGETS_DIR}/targets.pt" ]; then
    log_phase "Stage A: generate iter-2 VQ targets (parallel-VQ, K1=${K1} K2=${K2})"
    python -m CausalSpecUnit.generate_iter2_vq_targets \
        --data-root "${DATA_ROOT}" \
        --splits train-clean-100 train-clean-360 train-other-500 \
        --cmvn-path "${SOURCE_TARGETS_DIR}/cmvn.pt" \
        --ssl-checkpoint "${ITER1_SSL_CKPT}" \
        --output-dir "${ITER2_TARGETS_DIR}" \
        --variant xs \
        --chunk-size 8 --chunk-stride 4 --pca-dim 64 \
        --quantizer-type "${QUANTIZER_TYPE}" \
        --K1 "${K1}" --K2 "${K2}" \
        --beta 0.25 --decay 0.99 \
        --vq-steps 30000 --vq-batch-size 8192 \
        --max-fit-frames 1000000 \
        --fit-frames-per-batch 8192 \
        --batch-size 32 \
        --workers 8 \
        --dataloader-timeout 180 \
        --target-shards 128 \
        --seed 42
else
    log_phase "Stage A: SKIP (iter-2 targets.pt already exists)"
fi

# ==========================================================
# Stage B: Shard targets (generate_iter2_vq_targets.py already does this,
#          but double-check the index file landed)
# ==========================================================
if [ ! -f "${ITER2_TARGETS_DIR}/target_index.json" ]; then
    log_phase "Stage B: shard targets"
    python -m CausalSpecUnit.shard_targets --targets-dir "${ITER2_TARGETS_DIR}" --num-shards 128
else
    log_phase "Stage B: SKIP (target_index.json already exists)"
fi

# ==========================================================
# Stage C: SSL pretrain iter-2 (init encoder from iter-1, 100k more steps)
# ==========================================================
ITER2_SSL_CKPT_DIR="${ITER2_SSL_DIR}/checkpoint_step$(printf '%06d' ${ITER2_MAX_STEPS})"
ITER2_SSL_CKPT="${ITER2_SSL_CKPT_DIR}/checkpoint.pt"
if [ ! -f "${ITER2_SSL_CKPT}" ]; then
    log_phase "Stage C: iter-2 SSL pretrain ${ITER2_MAX_STEPS} steps | init from ${ITER1_SSL_CKPT}"
    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="$((PORT_BASE + 1))" \
        -m CausalSpecUnit.pretrain_ssl \
        --data-root "${DATA_ROOT}" \
        --targets-dir "${ITER2_TARGETS_DIR}" \
        --output-dir "${ITER2_SSL_DIR}" \
        --init-encoder-checkpoint "${ITER1_SSL_CKPT}" \
        --variant xs \
        --epochs 1000 \
        --max-steps "${ITER2_MAX_STEPS}" \
        --batch-size 128 \
        --grad-accum-steps 1 \
        --codebook-mode both \
        --k-coarse "${K_COARSE}" \
        --k-fine "${K_FINE}" \
        --mask-prob 0.30 \
        --mask-length 10 \
        --chunk-size 8 \
        --chunk-stride 4 \
        --lr 1e-3 \
        --warmup-epochs "${ITER2_WARMUP_EPOCHS}" \
        --peak-epochs "${ITER2_PEAK_EPOCHS}" \
        --noam-decay-rate 1.0 \
        --max-grad-norm 1.0 \
        --max-safe-grad-norm 200.0 \
        --workers 12 \
        --dataloader-timeout 300 \
        --prefetch-factor 4 \
        --log-every 10 \
        --save-every 10 \
        --progress off
else
    log_phase "Stage C: SKIP (iter-2 SSL checkpoint already exists at ${ITER2_SSL_CKPT})"
fi

[ -f "${ITER2_SSL_CKPT}" ] || { echo "Missing iter-2 SSL checkpoint after pretrain: ${ITER2_SSL_CKPT}"; exit 1; }

# ==========================================================
# Stage D: CTC fine-tune on train-clean-100 (seed 42)
# ==========================================================
FT_OUT_DIR="${FT_OUTPUT_ROOT}/${SUBSET}/seed${SEED}"
mkdir -p "${FT_OUT_DIR}"
if [ ! -f "${FT_OUT_DIR}/checkpoint_best/checkpoint.pt" ]; then
    log_phase "Stage D: CTC fine-tune | subset=${SUBSET} seed=${SEED} batch=${FT_BATCH_SIZE} (eff. $((FT_BATCH_SIZE * NPROC_PER_NODE)))"
    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="$((PORT_BASE + 2))" \
        -m CausalSpecUnit.train_ctc \
        --data-root "${DATA_ROOT}" \
        --cmvn-path "${SOURCE_TARGETS_DIR}/cmvn.pt" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --train-splits "${SUBSET}" \
        --ssl-checkpoint "${ITER2_SSL_CKPT}" \
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
    log_phase "Stage D: SKIP (best CTC checkpoint already exists)"
fi

# ==========================================================
# Stage E: Evaluate
# ==========================================================
if [ ! -f "${FT_OUT_DIR}/eval_results.json" ]; then
    log_phase "Stage E: evaluate test-clean + test-other"
    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="$((PORT_BASE + 3))" \
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
else
    log_phase "Stage E: SKIP (eval_results.json already exists)"
fi

log_phase "DONE VQ iter-2 (parallel) 100h"
echo "Iter-1 SSL:          ${ITER1_SSL_CKPT}"
echo "Iter-2 VQ targets:   ${ITER2_TARGETS_DIR}/targets.pt"
echo "Iter-2 SSL:          ${ITER2_SSL_CKPT}"
echo "Fine-tune (100h):    ${FT_OUT_DIR}/checkpoint_best/checkpoint.pt"
echo "Eval results:        ${FT_OUT_DIR}/eval_results.json"
echo ""
echo "Compare against:"
echo "  iter-1 parallel-VQ 100h: 16.22 / 35.36 test-clean/test-other"
echo "  iter-2 dual k-means 100h: 16.10 / 36.20"
