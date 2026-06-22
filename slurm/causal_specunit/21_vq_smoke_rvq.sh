#!/bin/bash
# VQ smoke test: RVQ (residual VQ, K1=100, K2=500) on SqueezeFormer-XS.
#
# Self-contained: from a trained RVQ quantizer through to test-clean/test-other
# WER numbers. Stages are gated by output-file existence so a re-submission
# resumes from wherever the previous run left off.
#
# Pipeline:
#   A. Generate VQ targets (CausalSpecUnit.generate_vq_targets)
#   B. Shard targets (CausalSpecUnit.shard_targets)
#   C. SSL pretrain 150k steps on RVQ targets
#   D. Fine-tune on Libri-Light 10h, seed 42 (one cell smoke)
#   E. Evaluate on test-clean / test-other
#
# Prerequisite: outputs/causal_specunit/vq/rvq_100_500/state.pt must exist.
# Run slurm/causal_specunit/20_train_quantizer.sh with QUANTIZER_TYPE=rvq first.
#
# Submit:
#   sbatch slurm/causal_specunit/21_vq_smoke_rvq.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=vq_rvq
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/vq_rvq.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/vq_rvq.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

# -------------------- Quantizer-specific config --------------------
QUANTIZER_TYPE="rvq"
QUANTIZER_DIR="outputs/causal_specunit/vq/rvq_100_500"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c8_rvq_100_500"
SSL_OUTPUT_DIR="outputs/causal_specunit/ssl_rvq_iter1_150k"
FT_OUTPUT_ROOT="outputs/causal_specunit/vq_smoke/rvq"
CODEBOOK_MODE="both"
K_COARSE="100"
K_FINE="500"

# -------------------- Shared config --------------------
PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
SOURCE_TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
TOKENIZER_PATH="dataset/bpe128.model"
NPROC_PER_NODE="2"

MAX_STEPS="${MAX_STEPS:-150000}"
PEAK_EPOCHS="${PEAK_EPOCHS:-30}"
SUBSET="${SUBSET:-librilight_10h}"
SEED="${SEED:-42}"
FT_BATCH_SIZE="${FT_BATCH_SIZE:-64}"   # 2 GPUs * 64 = 128 effective, matches script 10's 10h cell
FT_EPOCHS="${FT_EPOCHS:-150}"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONFAULTHANDLER_TIMEOUT=300
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
mkdir -p logs "${TARGETS_DIR}" "${SSL_OUTPUT_DIR}" "${FT_OUTPUT_ROOT}"

# -------------------- Sanity checks --------------------
[ -d "${VIRTUAL_ENV}" ]                          || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}" ]                            || { echo "Missing data root: ${DATA_ROOT}"; exit 1; }
[ -f "${SOURCE_TARGETS_DIR}/cluster_artifacts.joblib" ] || { echo "Missing source artifacts"; exit 1; }
[ -f "${SOURCE_TARGETS_DIR}/cmvn.pt" ]           || { echo "Missing source CMVN"; exit 1; }
[ -f "${QUANTIZER_DIR}/state.pt" ]               || { echo "Missing quantizer: ${QUANTIZER_DIR}/state.pt. Run 20_train_quantizer.sh with QUANTIZER_TYPE=rvq first."; exit 1; }
[ -f "${TOKENIZER_PATH}" ]                       || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }

log_phase() {
    echo ""
    echo "===================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "===================================================="
}

PORT_BASE=$((35000 + (SLURM_JOB_ID % 2500)))

# ==========================================================
# Stage A: Generate VQ targets
# ==========================================================
if [ ! -f "${TARGETS_DIR}/targets.pt" ]; then
    log_phase "Stage A: generate VQ targets (quantizer=${QUANTIZER_TYPE})"
    python -m CausalSpecUnit.generate_vq_targets \
        --source-targets-dir "${SOURCE_TARGETS_DIR}" \
        --quantizer-dir "${QUANTIZER_DIR}" \
        --output-dir "${TARGETS_DIR}" \
        --data-root "${DATA_ROOT}"
else
    log_phase "Stage A: SKIP (targets.pt already exists)"
fi

# ==========================================================
# Stage B: Shard targets so pretrain_ssl's per-worker loader can mmap them
# ==========================================================
if [ ! -f "${TARGETS_DIR}/target_index.json" ]; then
    log_phase "Stage B: shard targets"
    python -m CausalSpecUnit.shard_targets --targets-dir "${TARGETS_DIR}" --num-shards 128
else
    log_phase "Stage B: SKIP (target_index.json already exists)"
fi

# ==========================================================
# Stage C: SSL pretrain on VQ targets
# ==========================================================
SSL_CKPT_DIR="${SSL_OUTPUT_DIR}/checkpoint_step$(printf '%06d' ${MAX_STEPS})"
SSL_CKPT="${SSL_CKPT_DIR}/checkpoint.pt"
if [ ! -f "${SSL_CKPT}" ]; then
    log_phase "Stage C: SSL pretrain ${MAX_STEPS} steps | codebook_mode=${CODEBOOK_MODE} K_c=${K_COARSE} K_f=${K_FINE}"
    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="$((PORT_BASE + 1))" \
        -m CausalSpecUnit.pretrain_ssl \
        --data-root "${DATA_ROOT}" \
        --targets-dir "${TARGETS_DIR}" \
        --output-dir "${SSL_OUTPUT_DIR}" \
        --variant xs \
        --epochs 1000 \
        --max-steps "${MAX_STEPS}" \
        --batch-size 128 \
        --grad-accum-steps 1 \
        --codebook-mode "${CODEBOOK_MODE}" \
        --k-coarse "${K_COARSE}" \
        --k-fine "${K_FINE}" \
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
        --workers 12 \
        --dataloader-timeout 300 \
        --prefetch-factor 4 \
        --log-every 10 \
        --save-every 10 \
        --progress off
else
    log_phase "Stage C: SKIP (SSL checkpoint already exists at ${SSL_CKPT})"
fi

[ -f "${SSL_CKPT}" ] || { echo "Missing SSL checkpoint after pretrain: ${SSL_CKPT}"; exit 1; }

# ==========================================================
# Stage D: CTC fine-tune (one cell: ${SUBSET}, seed=${SEED})
# ==========================================================
FT_OUT_DIR="${FT_OUTPUT_ROOT}/${SUBSET}/seed${SEED}"
mkdir -p "${FT_OUT_DIR}"
if [ ! -f "${FT_OUT_DIR}/checkpoint_best/checkpoint.pt" ]; then
    log_phase "Stage D: CTC fine-tune | subset=${SUBSET} seed=${SEED} batch=${FT_BATCH_SIZE} epochs=${FT_EPOCHS}"
    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="$((PORT_BASE + 2))" \
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
    log_phase "Stage D: SKIP (best CTC checkpoint already exists)"
fi

# ==========================================================
# Stage E: Evaluate on test-clean / test-other
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

log_phase "DONE VQ smoke (${QUANTIZER_TYPE})"
echo "Quantizer:       ${QUANTIZER_DIR}/state.pt"
echo "VQ targets:      ${TARGETS_DIR}/targets.pt"
echo "SSL checkpoint:  ${SSL_CKPT}"
echo "Fine-tune:       ${FT_OUT_DIR}/checkpoint_best/checkpoint.pt"
echo "Eval results:    ${FT_OUT_DIR}/eval_results.json"
