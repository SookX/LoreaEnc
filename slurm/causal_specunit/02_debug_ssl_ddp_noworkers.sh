#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_dbg_ddp0w
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_dbg_ddp0w.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_dbg_ddp0w.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h"
OUTPUT_DIR="outputs/causal_specunit/debug_ssl_ddp_noworkers"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((15000 + SLURM_JOB_ID % 20000))}"
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

echo "2-GPU no-worker SSL debug starting at $(date)"
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Torchrun: $(which torchrun)"
echo "Targets: ${TARGETS_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

python - <<'PY'
import json
import os
import torch

targets_dir = "outputs/causal_specunit/targets_960h"
with open(os.path.join(targets_dir, "metadata.json"), encoding="utf-8") as f:
    metadata = json.load(f)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("metadata", {k: metadata.get(k) for k in ("chunk_size", "chunk_stride", "pca_dim", "num_target_utterances")})
PY

torchrun \
    --nproc_per_node=2 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m CausalSpecUnit.pretrain_ssl \
    --data-root "${DATA_ROOT}" \
    --targets-dir "${TARGETS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --variant xs \
    --epochs 1 \
    --max-train-batches 100 \
    --batch-size 128 \
    --grad-accum-steps 1 \
    --mask-prob 0.35 \
    --mask-length 10 \
    --chunk-size 4 \
    --chunk-stride 4 \
    --lr 1e-3 \
    --warmup-epochs 20 \
    --peak-epochs 20 \
    --noam-decay-rate 1.0 \
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 200.0 \
    --workers 0 \
    --dataloader-timeout 0 \
    --log-every 10 \
    --progress on

echo "2-GPU no-worker SSL debug finished at $(date)"
