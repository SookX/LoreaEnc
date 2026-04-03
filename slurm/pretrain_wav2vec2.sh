#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# wav2vec2 pre-training from scratch on 8×H200 (Discoverer)
# Submit: sbatch slurm/pretrain_wav2vec2.sh
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=wav2vec2_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8          # one task per GPU
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8            # 4 DataLoader workers × 2 for headroom
#SBATCH --mem=0                      # use all available node memory
#SBATCH --time=72:00:00
#SBATCH --output=logs/wav2vec2_pretrain_%j.out
#SBATCH --error=logs/wav2vec2_pretrain_%j.err
# Adjust partition / account to match your Discoverer allocation:
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT

mkdir -p logs outputs/wav2vec2_pretrained

# ── Environment ────────────────────────────────────────────────────────────
module purge
module load CUDA/12.1        # adjust to available CUDA module on Discoverer
module load Python/3.11      # adjust as needed

# NCCL tuning for InfiniBand (typical on Discoverer)
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

# ── Generate labeled splits once (skip if already done) ───────────────────
if [ ! -f dataset/splits/10h.json ]; then
    echo "Creating labeled split manifests …"
    python dataset/splits.py --data_root ./dataset --seed 42
fi

# ── Pre-train ──────────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    baselines/pretrain_wav2vec2.py \
        --data_root      ./dataset \
        --output_dir     ./outputs/wav2vec2_pretrained \
        --max_steps      400000 \
        --batch_size     32 \
        --lr             5e-4 \
        --warmup_steps   32000 \
        --precision      bf16 \
        --save_steps     10000 \
        --eval_steps     10000 \
        --log_steps      200 \
        --num_workers    4 \
        --seed           42
