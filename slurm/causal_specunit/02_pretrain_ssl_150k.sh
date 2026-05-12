#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ssl150k
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ssl150k.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ssl150k.%j.err

set -euo pipefail

echo "[$(date)] Job ${SLURM_JOB_ID} started on $(hostname)"
echo "[$(date)] Loading modules..."

module purge
module load anaconda3
module load nvidia/cuda/12

echo "[$(date)] Modules loaded."

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h"
OUTPUT_DIR="outputs/causal_specunit/pretrain_ssl_150k"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

echo "[$(date)] PROJECT_DIR=${PROJECT_DIR}"
echo "[$(date)] Python: $(which python)"
echo "[$(date)] Torchrun: $(which torchrun)"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

# --- Pre-flight checks ---
echo "[$(date)] Running pre-flight checks..."

if [ ! -f "${TARGETS_DIR}/targets.pt" ]; then
    echo "[ERROR] Missing targets: ${TARGETS_DIR}/targets.pt"
    echo "[ERROR] Run slurm/causal_specunit/01_generate_targets.sh first."
    exit 1
fi
echo "[$(date)] targets.pt OK ($(du -sh ${TARGETS_DIR}/targets.pt | cut -f1))"

if [ ! -f "${TARGETS_DIR}/cmvn.pt" ]; then
    echo "[ERROR] Missing CMVN: ${TARGETS_DIR}/cmvn.pt"
    exit 1
fi
echo "[$(date)] cmvn.pt OK"

if [ ! -f "${TARGETS_DIR}/metadata.json" ]; then
    echo "[ERROR] Missing metadata: ${TARGETS_DIR}/metadata.json"
    exit 1
fi
echo "[$(date)] metadata.json OK: $(cat ${TARGETS_DIR}/metadata.json)"

if [ ! -d "${DATA_ROOT}" ]; then
    echo "[ERROR] Missing data root: ${DATA_ROOT}"
    exit 1
fi
echo "[$(date)] Data root OK"

echo "[$(date)] Pre-flight checks passed."

# --- Environment ---
export MASTER_ADDR="${MASTER_ADDR:-$(hostname -s)}"
export MASTER_PORT="${MASTER_PORT:-$((13000 + SLURM_JOB_ID % 20000))}"
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

NUM_PROCESSES=2
WORKERS=12
DATALOADER_TIMEOUT=120

echo "[$(date)] Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "[$(date)] Processes: ${NUM_PROCESSES}  Workers: ${WORKERS}"

# --- CUDA / PyTorch check ---
python - <<'PY'
import sys, torch
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}  {p.total_memory // 1024**3}GB")
PY

echo "[$(date)] Starting torchrun..."

torchrun \
    --nproc_per_node="${NUM_PROCESSES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m CausalSpecUnit.pretrain_ssl \
    --data-root "${DATA_ROOT}" \
    --targets-dir "${TARGETS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --variant xs \
    --epochs 1000 \
    --max-steps 150000 \
    --batch-size 128 \
    --grad-accum-steps 1 \
    --mask-prob 0.065 \
    --mask-length 10 \
    --chunk-size 4 \
    --chunk-stride 4 \
    --lr 1e-3 \
    --warmup-epochs 20 \
    --peak-epochs 20 \
    --noam-decay-rate 1.0 \
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 200.0 \
    --workers "${WORKERS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --log-every 500 \
    --save-every 5 \
    --progress on

echo "[$(date)] Job ${SLURM_JOB_ID} SSL pretraining finished successfully."
