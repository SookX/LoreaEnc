#!/bin/bash
#SBATCH --job-name=sqformer_baseline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --mail-type=END,FAIL

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p logs outputs/baseline/1h outputs/baseline/100h outputs/baseline/960h
echo $SLURM_JOB_ID > logs/baseline_jobid.txt

source activate Torch

export MASTER_ADDR=localhost
export MASTER_PORT=12355
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# accelerate needs to know how many GPUs to use on this node
export NUM_PROCESSES=2

# ── Label fraction consistency note ───────────────────────────────────────────
# All four ablation models (baseline, patches, SSL-InfoNCE, SSL-InfoNCE+cosine)
# train on IDENTICAL data splits per label fraction. --seed 42 and --hours
# passed to train.py guarantee reproducible subsampling via subsample_by_hours().
# All models use the same full train_960 splits; sampling happens inside train.py.
# Do NOT create separate manifest files per run.

# ── Run 1: 960 hours full dataset ─────────────────────────────────────────────
echo "Starting run: 960h $(date)"
accelerate launch \
  --num_processes $NUM_PROCESSES \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  SqueezeFormer/train.py \
    --hours 960 \
    --seed 42 \
    --output-dir outputs/baseline/960h \
    --run-name baseline_960h \
    --workers 8
if [ $? -ne 0 ]; then
  echo "ERROR: 960h run failed. Stopping."
  exit 1
fi
echo "Completed run: 960h $(date)"

# ── Run 2: 100 hours of labeled data ──────────────────────────────────────────
echo "Starting run: 100h $(date)"
accelerate launch \
  --num_processes $NUM_PROCESSES \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  SqueezeFormer/train.py \
    --hours 100 \
    --seed 42 \
    --output-dir outputs/baseline/100h \
    --run-name baseline_100h \
    --workers 8
if [ $? -ne 0 ]; then
  echo "ERROR: 100h run failed. Stopping."
  exit 1
fi
echo "Completed run: 100h $(date)"

# ── Run 3: 1 hour of labeled data ─────────────────────────────────────────────
echo "Starting run: 1h $(date)"
accelerate launch \
  --num_processes $NUM_PROCESSES \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  SqueezeFormer/train.py \
    --hours 1 \
    --seed 42 \
    --output-dir outputs/baseline/1h \
    --run-name baseline_1h \
    --workers 8
if [ $? -ne 0 ]; then
  echo "ERROR: 1h run failed. Stopping."
  exit 1
fi
echo "Completed run: 1h $(date)"

# ── Done ──────────────────────────────────────────────────────────────────────
echo "All baseline runs completed successfully $(date)"
