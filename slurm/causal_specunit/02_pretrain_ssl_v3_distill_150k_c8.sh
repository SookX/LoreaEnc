#!/bin/bash
# v3 SSL pretraining: v2 recipe + data2vec-style EMA-teacher distillation.
#
# Adds ONE new mechanism on top of v2:
#   The student must additionally predict, at masked positions, the *continuous*
#   encoder features of an EMA teacher that processed the **clean** mel. The
#   teacher is a non-trainable copy of the encoder updated each optimizer step
#   with momentum 0.999.
#
# Why this is strictly an SSL improvement:
#   The teacher only exists because the encoder is self-supervising. There is
#   no analog in supervised CTC scratch training — an EMA of a randomly-init
#   model is uninformative for the first ~30 epochs, so trying to apply this
#   to scratch would inject noise. SSL has an encoder being pretrained on the
#   same domain; the EMA of it is meaningful from step ~1k onward.
#
# Why it improves quality (vs v2 alone):
#   v2 only supervises the encoder against 9-bit cluster IDs (K=500). The
#   encoder's 144-dim feature is 144 floats per position — 100x the bandwidth
#   that cluster CE uses. EMA-teacher distillation uses *all* of it as a
#   continuous target. Loss: smooth-L1 on layer-normed features at masked
#   positions. Combined as: cluster_CE + aux_weight*aux_CE + distill_weight*L1.
#
# Cost: teacher is ~9M params (deep copy of encoder, no_grad). Extra forward
# pass under no_grad on clean mel. Roughly +25% wall clock per step vs v2.
# Combined with 150k steps: ~11-13h on 4 GPUs.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ssl_v3_distill_150k_c8
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ssl_v3_distill_150k_c8.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ssl_v3_distill_150k_c8.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/pretrain_ssl_v3_distill_150k_c8}"

if [ ! -d "${VIRTUAL_ENV}" ]; then
    echo "Missing venv: ${VIRTUAL_ENV}"
    exit 1
fi

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

[ -d "${DATA_ROOT}" ]                  || { echo "Missing data: ${DATA_ROOT}"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]        || { echo "Missing CMVN"; exit 1; }
[ -f "${TARGETS_DIR}/targets.pt" ]     || { echo "Missing targets"; exit 1; }
[ -f "${TARGETS_DIR}/metadata.json" ]  || { echo "Missing target metadata"; exit 1; }
if [ ! -f "${TARGETS_DIR}/target_index.json" ]; then
    python -m CausalSpecUnit.shard_targets --targets-dir "${TARGETS_DIR}" --num-shards 128
fi

# v2 recipe knobs (unchanged)
MASK_PROB="${MASK_PROB:-0.50}"
MASK_LENGTH="${MASK_LENGTH:-10}"
LAYER_DROP_P="${LAYER_DROP_P:-0.05}"
AUX_LAYERS="${AUX_LAYERS:-8}"
AUX_WEIGHT="${AUX_WEIGHT:-0.5}"
CODEBOOK_MODE="${CODEBOOK_MODE:-both}"

# v3 distillation knobs
TEACHER_DISTILL_WEIGHT="${TEACHER_DISTILL_WEIGHT:-0.5}"
TEACHER_MOMENTUM="${TEACHER_MOMENTUM:-0.999}"

MAX_STEPS="${MAX_STEPS:-150000}"
NUM_PROCESSES=4
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
WORKERS="${WORKERS:-8}"
DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-300}"

export TARGETS_DIR
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((18000 + SLURM_JOB_ID % 20000))}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "Job ${SLURM_JOB_ID} v3 SSL pretraining (v2 + EMA distillation) starting at $(date)"
echo "Output:                       ${OUTPUT_DIR}"
echo "Max steps:                    ${MAX_STEPS}"
echo "Effective batch:              $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
echo "v2 recipe:                    mask_prob=${MASK_PROB} layer_drop=${LAYER_DROP_P} aux_layers=${AUX_LAYERS} aux_weight=${AUX_WEIGHT}"
echo "v3 EMA distillation:          weight=${TEACHER_DISTILL_WEIGHT} momentum=${TEACHER_MOMENTUM} (norm_targets=on, smooth_l1)"
echo "Codebook objective:           ${CODEBOOK_MODE}"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY

RESUME_CKPT="${RESUME_CKPT:-}"
if [ -n "${RESUME_CKPT}" ]; then
    [ -f "${RESUME_CKPT}/checkpoint.pt" ] || { echo "RESUME_CKPT bad: ${RESUME_CKPT}"; exit 1; }
fi

read -r -a AUX_LAYERS_ARGS <<< "${AUX_LAYERS}"

torchrun \
    --nproc_per_node="${NUM_PROCESSES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m CausalSpecUnit.pretrain_ssl \
    --data-root "${DATA_ROOT}" \
    --targets-dir "${TARGETS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --variant xs \
    --epochs 1000 \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --codebook-mode "${CODEBOOK_MODE}" \
    --mask-prob "${MASK_PROB}" \
    --mask-length "${MASK_LENGTH}" \
    --chunk-size 8 \
    --chunk-stride 4 \
    --layer-drop-p "${LAYER_DROP_P}" \
    --aux-layers "${AUX_LAYERS_ARGS[@]}" \
    --aux-weight "${AUX_WEIGHT}" \
    --specaug \
    --specaug-time-mask-param 30 \
    --specaug-freq-mask-param 20 \
    --specaug-time-masks 2 \
    --specaug-freq-masks 2 \
    --teacher-distill-weight "${TEACHER_DISTILL_WEIGHT}" \
    --teacher-momentum "${TEACHER_MOMENTUM}" \
    --teacher-normalize-targets \
    --lr 1e-3 \
    --warmup-epochs 20 \
    --peak-epochs 20 \
    --noam-decay-rate 1.0 \
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 200.0 \
    --workers "${WORKERS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --prefetch-factor 4 \
    --log-every 100 \
    --save-every 5 \
    --trace-startup \
    --progress off \
    $( [ -n "${RESUME_CKPT}" ] && echo "--resume ${RESUME_CKPT}" || true )

echo "Job ${SLURM_JOB_ID} v3 SSL pretraining finished at $(date)"
echo "Output checkpoint: ${OUTPUT_DIR}/checkpoint_step$(printf '%06d' ${MAX_STEPS})/checkpoint.pt"
