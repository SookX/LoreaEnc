#!/bin/bash
# SqueezeFormer-M smoke test: 10k SSL steps on the existing dual-kmeans
# targets, just to verify the M variant trains stably and to measure
# per-step throughput so we can budget for the full 400k iter-1 run.
#
# Why this exists:
#   - SqueezeFormer-M (encoder_dim=324, num_encoder_layers=20, ~75M params)
#     is ~8x the parameter count of XS, and per-step compute scales ~5-7x.
#   - Before committing ~200 H200-hours to a 400k iter-1 run, we want to
#     confirm:
#       1. Training is stable (no OOM, no NaN, loss decreases)
#       2. Actual per-step wall time (extrapolate to full 400k budget)
#       3. Batch size that fits on 2x H200 with reasonable accumulation
#
# What it does NOT test:
#   - Whether parallel-VQ targets work at M scale (defer until we know M
#     trains at all)
#   - Whether PCA-64 is the right feature dim (defer to a follow-up sweep)
#   - Anything iter-2 related
#
# Recipe matches iter-1 (150k) except:
#   - --variant m       (was xs)
#   - --max-steps 10000  (was 150000)
#   - --batch-size 64 + grad-accum 2  (preserves effective batch 256;
#                                       safer for memory at the larger scale)
#   - Output dir / log group differs so we don't clobber XS checkpoints
#
# Submit:
#   sbatch slurm/causal_specunit/30_pretrain_ssl_m_smoke_10k.sh
#
# Optional knobs:
#   MAX_STEPS=20000 sbatch ...   # longer smoke
#   BATCH_SIZE=128 sbatch ...    # try bigger batch if first run had headroom

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=ssl_m_smk
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/ssl_m_smk.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/ssl_m_smk.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/ssl_m_smoke_10k}"

MAX_STEPS="${MAX_STEPS:-10000}"
PEAK_EPOCHS="${PEAK_EPOCHS:-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"

# Sanity checks
[ -d "${VIRTUAL_ENV}" ]                          || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}" ]                            || { echo "Missing data root: ${DATA_ROOT}"; exit 1; }
[ -f "${TARGETS_DIR}/targets.pt" ]               || { echo "Missing targets: ${TARGETS_DIR}/targets.pt"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]                  || { echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"; exit 1; }
[ -f "${TARGETS_DIR}/metadata.json" ]            || { echo "Missing metadata: ${TARGETS_DIR}/metadata.json"; exit 1; }

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
export MASTER_PORT="${MASTER_PORT:-$((43000 + SLURM_JOB_ID % 20000))}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
mkdir -p logs "${OUTPUT_DIR}"

NUM_PROCESSES=2
WORKERS=12
DATALOADER_TIMEOUT=300

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SqueezeFormer-M smoke test"
echo "  variant=m  max_steps=${MAX_STEPS}  batch=${BATCH_SIZE}  grad_accum=${GRAD_ACCUM_STEPS}"
echo "  effective batch = ${BATCH_SIZE} x ${NUM_PROCESSES} x ${GRAD_ACCUM_STEPS} = $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
echo "  targets=${TARGETS_DIR}"
echo "  output=${OUTPUT_DIR}"
echo "===================================================="
echo ""

python - <<'PY'
import json, os, torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} = {props.name} | mem={props.total_memory/1e9:.1f} GB")

# Construct the M variant once to print the actual parameter count so we
# can compare against the 9M XS baseline.
from CausalSpecUnit.model import CausalSpecUnitSSL
m = CausalSpecUnitSSL(variant="m")
n = sum(p.numel() for p in m.parameters())
print(f"SqueezeFormer-M total params: {n/1e6:.2f}M")
print(f"  encoder params: {sum(p.numel() for p in m.encoder.parameters())/1e6:.2f}M")
PY

T0=$(date +%s)

torchrun \
    --nproc_per_node="${NUM_PROCESSES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m CausalSpecUnit.pretrain_ssl \
    --data-root "${DATA_ROOT}" \
    --targets-dir "${TARGETS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --variant m \
    --epochs 1000 \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --codebook-mode both \
    --k-coarse 100 \
    --k-fine 500 \
    --mask-prob 0.30 \
    --mask-length 10 \
    --chunk-size 8 \
    --chunk-stride 4 \
    --lr 1e-3 \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --peak-epochs "${PEAK_EPOCHS}" \
    --noam-decay-rate 1.0 \
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 200.0 \
    --workers "${WORKERS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --prefetch-factor 4 \
    --log-every 10 \
    --save-every 10 \
    --trace-startup \
    --progress on

T1=$(date +%s)
ELAPSED=$((T1 - T0))

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SqueezeFormer-M smoke DONE"
echo "  elapsed: ${ELAPSED} s = $((ELAPSED / 60)) min"
echo "  steps:   ${MAX_STEPS}"
echo "  per-step throughput: $(python3 -c "print(f'{${ELAPSED} / ${MAX_STEPS} * 1000:.1f} ms/step')")"
echo "===================================================="
echo ""
echo "Extrapolation to full iter-1 (400k steps):"
python3 - <<PY
elapsed_smoke = ${ELAPSED}
steps_smoke = ${MAX_STEPS}
target_steps = 400000
per_step = elapsed_smoke / steps_smoke
total_s = per_step * target_steps
print(f"  400k steps would take {total_s/3600:.1f} wall-hours on 2 H200s")
print(f"  = {total_s/3600 * 2:.1f} H200-hours")
PY
echo "===================================================="

echo "Output checkpoint: ${OUTPUT_DIR}/checkpoint_step$(printf '%06d' ${MAX_STEPS})/checkpoint.pt"
