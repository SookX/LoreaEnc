#!/bin/bash
# Run locally on a single GPU without Slurm for fast debugging.
# Usage: bash slurm/causal_specunit/02_debug_ssl_local.sh
set -euo pipefail

DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c2"
OUTPUT_DIR="outputs/causal_specunit/debug_ssl_local"
MEL_CACHE_DIR="outputs/causal_specunit/mel_cache_960h"

export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export CUDA_LAUNCH_BLOCKING=1   # synchronous CUDA — gives exact line numbers on errors

mkdir -p "${OUTPUT_DIR}"

echo "=== env check ==="
python - <<'PY'
import torch, json, os
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
targets_dir = "outputs/causal_specunit/targets_960h_c2"
with open(os.path.join(targets_dir, "metadata.json"), encoding="utf-8") as f:
    meta = json.load(f)
print("target metadata:", {k: meta.get(k) for k in
      ["chunk_size", "chunk_stride", "pca_dim", "k_coarse", "k_fine", "num_target_utterances"]})
PY

echo ""
echo "=== single-GPU pretrain (2 batches, no compile, workers=0) ==="
python -m CausalSpecUnit.pretrain_ssl \
    --data-root "${DATA_ROOT}" \
    --targets-dir "${TARGETS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --mel-cache-dir "${MEL_CACHE_DIR}" \
    --variant xs \
    --epochs 2 \
    --max-train-batches 2 \
    --batch-size 8 \
    --grad-accum-steps 1 \
    --mask-prob 0.40 \
    --mask-length 12 \
    --chunk-size 2 \
    --chunk-stride 4 \
    --lr 1e-3 \
    --warmup-epochs 1 \
    --peak-epochs 1 \
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 200.0 \
    --workers 0 \
    --dataloader-timeout 0 \
    --log-every 1 \
    --save-every 1 \
    --trace-startup \
    --progress on

echo "=== done ==="
