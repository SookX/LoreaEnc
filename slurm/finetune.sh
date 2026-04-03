#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CTC fine-tuning on 8×H200 (Discoverer)
# Run this three times (1h / 10h / 100h) for each model to produce the full
# set of baseline numbers for comparison with Lorea.
#
# Usage:
#   sbatch slurm/finetune.sh wav2vec2 ./outputs/wav2vec2_pretrained/checkpoint-400000 10h
#   sbatch slurm/finetune.sh hubert   ./outputs/hubert_pretrained/checkpoint-400000   10h
#
# Args: $1=model (wav2vec2|hubert)  $2=checkpoint  $3=label_budget (1h|10h|100h)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=finetune_%1_%3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/finetune_%x_%j.out
#SBATCH --error=logs/finetune_%x_%j.err
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT

MODEL=${1:-wav2vec2}
CHECKPOINT=${2:-./outputs/wav2vec2_pretrained/checkpoint-400000}
LABEL_BUDGET=${3:-10h}
MANIFEST="./dataset/splits/${LABEL_BUDGET}.json"
OUTPUT_DIR="./outputs/${MODEL}_finetuned_${LABEL_BUDGET}"

echo "Model:      ${MODEL}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Labels:     ${LABEL_BUDGET}  (${MANIFEST})"
echo "Output:     ${OUTPUT_DIR}"

mkdir -p logs "${OUTPUT_DIR}"

module purge
module load CUDA/12.1
module load Python/3.11

export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

# Step budget depends on label set size:
#   100h → 20 000 steps  (~20 epochs at effective batch 256)
#    10h →  8 000 steps
#     1h →  3 000 steps
if   [ "${LABEL_BUDGET}" = "100h" ]; then MAX_STEPS=20000
elif [ "${LABEL_BUDGET}" = "10h"  ]; then MAX_STEPS=8000
else                                       MAX_STEPS=3000
fi

torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    baselines/finetune.py \
        --model          "${MODEL}" \
        --checkpoint     "${CHECKPOINT}" \
        --train_manifest "${MANIFEST}" \
        --output_dir     "${OUTPUT_DIR}" \
        --max_steps      "${MAX_STEPS}" \
        --batch_size     32 \
        --lr             1e-4 \
        --warmup_ratio   0.1 \
        --precision      bf16 \
        --save_steps     1000 \
        --eval_steps     1000 \
        --log_steps      100 \
        --num_workers    4 \
        --seed           42
