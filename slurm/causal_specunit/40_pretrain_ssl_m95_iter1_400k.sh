#!/bin/bash
# SqueezeFormer-M95 iter-1 SSL pretrain at the wav2vec2-Base / HuBERT-Base
# matched scale (93M params, ~98% of the 95M baseline param count).
#
# 400k optimizer steps to match wav2vec2-Base's exact pretrain budget.
# Reuses the existing dual-kmeans targets at 960h, K=100+500. Iter-2 will
# be a separate run after this finishes.
#
# Throughput estimate (extrapolated from M smoke at 55.8M, 672.8 ms/step,
# scaled by ~1.7x for the 93M parameter count + 24 layers vs 20):
#   ~1.0-1.2 s per step on 2 H200s
#   400k steps => ~120-130 wall-hours = ~240-260 H200-hours
#   Walltime requested below assumes the slow end of that range plus buffer.
#
# Submit:
#   sbatch slurm/causal_specunit/40_pretrain_ssl_m95_iter1_400k.sh
#
# Optional knobs:
#   BATCH_SIZE=128 GRAD_ACCUM_STEPS=1 sbatch ...    # if memory has headroom
#   MAX_STEPS=200000 sbatch ...                    # shorter run

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=m95_i1_400k
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/m95_i1_400k.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/m95_i1_400k.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
# Parallel-VQ targets (K1=100, K2=500), the recipe that beat k-means at 9M.
# k-means (targets_960h_c8) has a low information ceiling that caps the
# encoder regardless of scale, so the 95M run uses the learned-VQ targets.
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8_parallel_100_500}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/ssl_m95_parallelvq_iter1_400k}"

MAX_STEPS="${MAX_STEPS:-400000}"
LR="${LR:-5e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-30}"
PEAK_EPOCHS="${PEAK_EPOCHS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-32000}"
PEAK_STEPS="${PEAK_STEPS:-22000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
# Codebook sizes — must match the targets' metadata.json. Parallel-VQ here is
# 100+500. Do NOT scale these with model size for iter-1: the codebook is a
# property of the PCA-64 chunk feature space, not the encoder (HuBERT keeps
# K=500 from Base to X-Large). Scale the fine codebook at iter-2 instead,
# where targets come from the richer 95M hidden states.
K_COARSE="${K_COARSE:-100}"
K_FINE="${K_FINE:-500}"
# Masking strength. wav2vec2/HuBERT mask ~45-50% of frames; the formula
# n_spans = mask_prob*T/mask_length with mask_prob=0.30 only masked ~19%.
# 0.65 brings the masked fraction into the standard SSL range.
MASK_PROB="${MASK_PROB:-0.65}"
MASK_LENGTH="${MASK_LENGTH:-10}"

# Sanity checks
[ -d "${VIRTUAL_ENV}" ]                          || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}" ]                            || { echo "Missing data root: ${DATA_ROOT}"; exit 1; }
[ -f "${TARGETS_DIR}/targets.pt" ]               || { echo "Missing targets: ${TARGETS_DIR}/targets.pt"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]                  || { echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"; exit 1; }
[ -f "${TARGETS_DIR}/metadata.json" ]            || { echo "Missing metadata: ${TARGETS_DIR}/metadata.json"; exit 1; }

# Fail fast if K_COARSE/K_FINE disagree with the targets' metadata. pretrain_ssl
# validates this too, but catching it here avoids burning a slurm allocation
# only to die on the first batch.
python3 - "$TARGETS_DIR" "$K_COARSE" "$K_FINE" <<'PY'
import json, sys
targets_dir, k_coarse, k_fine = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(f"{targets_dir}/metadata.json") as f:
    md = json.load(f)
errs = []
if int(md.get("k_coarse", -1)) != k_coarse:
    errs.append(f"k_coarse: targets={md.get('k_coarse')} script={k_coarse}")
if int(md.get("k_fine", -1)) != k_fine:
    errs.append(f"k_fine: targets={md.get('k_fine')} script={k_fine}")
if errs:
    print("Codebook size mismatch vs targets metadata:\n  " + "\n  ".join(errs))
    sys.exit(1)
print(f"Targets OK: k_coarse={md.get('k_coarse')} k_fine={md.get('k_fine')} "
      f"type={md.get('quantizer_type', 'kmeans')} utts={md.get('num_target_utterances')}")
PY

if [ ! -f "${TARGETS_DIR}/target_index.json" ]; then
    echo "Target shards index missing; building 128 shards..."
    python -m CausalSpecUnit.shard_targets --targets-dir "${TARGETS_DIR}" --num-shards 128
fi

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
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((44000 + SLURM_JOB_ID % 20000))}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
mkdir -p logs "${OUTPUT_DIR}"

NUM_PROCESSES=2
WORKERS=12
DATALOADER_TIMEOUT=300

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SqueezeFormer-M95 iter-1 SSL pretrain"
echo "  variant=m95 (93M params, ~98% of wav2vec2-Base)"
echo "  max_steps=${MAX_STEPS} batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM_STEPS} lr=${LR}"
echo "  targets=${TARGETS_DIR}  (K_c=${K_COARSE} K_f=${K_FINE})"
echo "  mask_prob=${MASK_PROB} mask_length=${MASK_LENGTH}"
echo "  schedule warmup_steps=${WARMUP_STEPS} peak_steps=${PEAK_STEPS}"
echo "  effective batch = ${BATCH_SIZE} x ${NUM_PROCESSES} x ${GRAD_ACCUM_STEPS} = $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
echo "  targets=${TARGETS_DIR}"
echo "  output=${OUTPUT_DIR}"
echo "===================================================="
echo ""

python - <<'PY'
import json, os, torch
print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} = {props.name} | mem={props.total_memory/1e9:.1f} GB")

from CausalSpecUnit.model import CausalSpecUnitSSL
m = CausalSpecUnitSSL(variant="m95")
n = sum(p.numel() for p in m.parameters())
ne = sum(p.numel() for p in m.encoder.parameters())
print(f"SqueezeFormer-M95: total={n/1e6:.2f}M  encoder={ne/1e6:.2f}M")
print(f"  cf. wav2vec2-Base: 95M  HuBERT-Base: 95M")
PY

T0=$(date +%s)

# 4h wall-time cap: resume from the latest checkpoint and queue a successor so
# the full 400k-step run completes across a chain of 4h jobs. Exits here if
# MAX_STEPS is already reached. Sets RESUME_CKPT, consumed by the block below.
SELF_SCRIPT="slurm/causal_specunit/40_pretrain_ssl_m95_iter1_400k.sh"
source slurm/causal_specunit/_autochain.sh

RESUME_CKPT="${RESUME_CKPT:-}"
RESUME_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
    if [ ! -f "${RESUME_CKPT}/checkpoint.pt" ] && [ ! -f "${RESUME_CKPT}" ]; then
        echo "RESUME_CKPT set but not found: ${RESUME_CKPT}"
        exit 1
    fi
    echo "Resuming from checkpoint: ${RESUME_CKPT}"
    RESUME_ARGS=(--resume "${RESUME_CKPT}")
fi

torchrun \
    --nproc_per_node="${NUM_PROCESSES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m CausalSpecUnit.pretrain_ssl \
    --data-root "${DATA_ROOT}" \
    --targets-dir "${TARGETS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --variant m95 \
    --epochs 1000 \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --codebook-mode both \
    --k-coarse "${K_COARSE}" \
    --k-fine "${K_FINE}" \
    --mask-prob "${MASK_PROB}" \
    --mask-length "${MASK_LENGTH}" \
    --chunk-size 8 \
    --chunk-stride 4 \
    --lr "${LR}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --peak-epochs "${PEAK_EPOCHS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --peak-steps "${PEAK_STEPS}" \
    --noam-decay-rate 1.0 \
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 200.0 \
    --workers "${WORKERS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --prefetch-factor 4 \
    --log-every 50 \
    --save-every 1 \
    --save-at-steps 150000 300000 \
    --keep-checkpoints 5 \
    --trace-startup \
    --progress off \
    "${RESUME_ARGS[@]}"

T1=$(date +%s)
ELAPSED=$((T1 - T0))

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] M95 iter-1 SSL DONE"
echo "  elapsed: ${ELAPSED} s = $((ELAPSED / 3600)) h $((ELAPSED % 3600 / 60)) m"
echo "  steps:   ${MAX_STEPS}"
python3 -c "print(f'  throughput: {${ELAPSED} / ${MAX_STEPS} * 1000:.1f} ms/step')"
echo "===================================================="
echo "Output checkpoint: ${OUTPUT_DIR}/checkpoint_step$(printf '%06d' ${MAX_STEPS})/checkpoint.pt"
