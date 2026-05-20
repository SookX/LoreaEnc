#!/bin/bash
# Improved iter-1 SSL pretraining ("v2" recipe).
#
# Four orthogonal training-side changes vs the original iter-1 recipe. None
# require regenerating targets — same targets_960h_c8 dir is consumed.
#
#   (1) Higher effective mask rate (0.30 → 0.50).
#       HuBERT-base and wav2vec2-base mask ~50% of frames effectively. The
#       original recipe under-masked relative to the literature, so the SSL
#       task was easier than intended and the encoder learned less per step.
#
#   (2) LayerDrop p=0.05 in the encoder during SSL.
#       Standard wav2vec2/HuBERT regularizer. Each SqueezeFormer block becomes
#       optional during training, which forces redundant feature extraction
#       across depth. The time-reduce and time-recover blocks are *never*
#       dropped (they change tensor shape).
#
#   (3) SpecAugment during SSL, filled with the *live* mask_emb token.
#       The fine-tune encoder must handle two corruption distributions: the
#       HuBERT mask_emb (seen during SSL) and SpecAug (seen only at fine-tune).
#       By applying SpecAug during SSL with the same mask_emb fill, the
#       encoder sees ONE consistent "masked" distribution; the gap at
#       fine-tune disappears.
#
#   (4) Auxiliary SSL prediction heads at the U-net bottleneck (layer 8).
#       SqueezeFormer XS has its time-reduction at idx 7. Block idx 8 is the
#       first half-rate block — the natural information bottleneck. Adding
#       k100/k500 prediction heads there forces the LOWER encoder stack to
#       be directly task-discriminative on the same targets, instead of
#       relying on a long backprop path from a single head at the top.
#       This is the novel piece of the recipe; the other three are standard
#       SSL practices we were missing.
#
# Targets, optimizer, LR schedule, mask_length, chunk_size/stride are unchanged.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ssl_v2_150k_c8
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ssl_v2_150k_c8.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ssl_v2_150k_c8.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/pretrain_ssl_v2_150k_c8}"

if [ ! -d "${VIRTUAL_ENV}" ]; then
    echo "Missing venv: ${VIRTUAL_ENV}"
    exit 1
fi

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${DATA_ROOT}" ]; then
    echo "Missing data root: ${DATA_ROOT}"; exit 1
fi
if [ ! -f "${TARGETS_DIR}/targets.pt" ]; then
    echo "Missing targets: ${TARGETS_DIR}/targets.pt"; exit 1
fi
if [ ! -f "${TARGETS_DIR}/cmvn.pt" ]; then
    echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"; exit 1
fi
if [ ! -f "${TARGETS_DIR}/metadata.json" ]; then
    echo "Missing metadata: ${TARGETS_DIR}/metadata.json"; exit 1
fi
if [ ! -f "${TARGETS_DIR}/target_index.json" ]; then
    echo "Missing sharded target index, creating..."
    python -m CausalSpecUnit.shard_targets --targets-dir "${TARGETS_DIR}" --num-shards 128
fi

export TARGETS_DIR

# Recipe knobs — overridable via environment, defaults match the v2 design
MASK_PROB="${MASK_PROB:-0.50}"
MASK_LENGTH="${MASK_LENGTH:-10}"
LAYER_DROP_P="${LAYER_DROP_P:-0.05}"
AUX_LAYERS="${AUX_LAYERS:-8}"
AUX_WEIGHT="${AUX_WEIGHT:-0.5}"
MAX_STEPS="${MAX_STEPS:-150000}"

# Effective batch unchanged: 4 GPUs × batch 64 × accum 1 = 256 (matches old 2×128×1).
# Keeps gradient noise comparable to the original SSL run for fair comparison.
NUM_PROCESSES=4
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
WORKERS=8
DATALOADER_TIMEOUT=300

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((14000 + SLURM_JOB_ID % 20000))}"
export PYTHONFAULTHANDLER=1
export PYTHONFAULTHANDLER_TIMEOUT=300
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "Job ${SLURM_JOB_ID} v2 SSL pretraining starting at $(date)"
echo "Python: $(which python)"
echo "Torchrun: $(which torchrun)"
echo "Data root: ${DATA_ROOT}"
echo "Targets: ${TARGETS_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Max steps: ${MAX_STEPS}"
echo "GPUs on node: ${NUM_PROCESSES}"
echo "Effective batch: $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
echo "v2 recipe: mask_prob=${MASK_PROB} (HuBERT-equivalent) layer_drop=${LAYER_DROP_P} aux_layers=${AUX_LAYERS} aux_weight=${AUX_WEIGHT} specaug=on"

python - <<'PY'
import json, os, torch
targets_dir = os.environ["TARGETS_DIR"]
with open(os.path.join(targets_dir, "metadata.json"), encoding="utf-8") as f:
    metadata = json.load(f)
print("PyTorch:", torch.__version__)
print("Target metadata:", {k: metadata.get(k) for k in
      ["chunk_size","chunk_stride","pca_dim","k_coarse","k_fine","target_features","num_target_utterances"]})
PY

RESUME_CKPT="${RESUME_CKPT:-}"
if [ -n "${RESUME_CKPT}" ]; then
    if [ ! -f "${RESUME_CKPT}/checkpoint.pt" ]; then
        echo "RESUME_CKPT set but checkpoint.pt not found: ${RESUME_CKPT}"; exit 1
    fi
    echo "Resuming from checkpoint: ${RESUME_CKPT}"
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
    --lr 1e-3 \
    --warmup-epochs 20 \
    --peak-epochs 20 \
    --noam-decay-rate 1.0 \
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 200.0 \
    --workers "${WORKERS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --prefetch-factor 4 \
    --log-every 10 \
    --save-every 10 \
    --trace-startup \
    --progress on \
    $( [ -n "${RESUME_CKPT}" ] && echo "--resume ${RESUME_CKPT}" || true )

echo "Job ${SLURM_JOB_ID} v2 SSL pretraining finished at $(date)"
echo "Output checkpoint: ${OUTPUT_DIR}"
