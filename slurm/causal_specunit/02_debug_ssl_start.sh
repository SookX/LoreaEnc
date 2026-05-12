#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_debug_ssl
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_debug_ssl.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_debug_ssl.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h"
OUTPUT_DIR="outputs/causal_specunit/debug_ssl"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

echo "Debug SSL job ${SLURM_JOB_ID} starting at $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Targets: ${TARGETS_DIR}"

python - <<'PY'
import json
import os
import time
import torch

targets_dir = "outputs/causal_specunit/targets_960h"
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
t = time.time()
with open(os.path.join(targets_dir, "metadata.json"), encoding="utf-8") as f:
    metadata = json.load(f)
print("metadata", {
    "chunk_size": metadata.get("chunk_size"),
    "chunk_stride": metadata.get("chunk_stride"),
    "pca_dim": metadata.get("pca_dim"),
    "target_features": metadata.get("target_features"),
    "num_target_utterances": metadata.get("num_target_utterances"),
})
print("metadata loaded in", round(time.time() - t, 2), "sec")
t = time.time()
targets = torch.load(os.path.join(targets_dir, "targets.pt"), map_location="cpu")
print("targets", len(targets), "loaded in", round(time.time() - t, 2), "sec")
PY

python -m CausalSpecUnit.pretrain_ssl \
    --data-root "${DATA_ROOT}" \
    --targets-dir "${TARGETS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --variant xs \
    --epochs 1 \
    --max-train-batches 2 \
    --batch-size 4 \
    --workers 0 \
    --dataloader-timeout 0 \
    --chunk-size 4 \
    --chunk-stride 4 \
    --mask-prob 0.35 \
    --mask-length 10 \
    --progress on

echo "Debug SSL job finished at $(date)"
