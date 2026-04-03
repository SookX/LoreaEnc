#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# HuBERT pre-training from scratch on 8×H200 (Discoverer)
# Submit: sbatch slurm/pretrain_hubert.sh
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=hubert_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=logs/hubert_pretrain_%j.out
#SBATCH --error=logs/hubert_pretrain_%j.err
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT

mkdir -p logs outputs/hubert_pretrained outputs/hubert_kmeans

module purge
module load CUDA/12.1
module load Python/3.11

export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

# ── Generate labeled splits once ──────────────────────────────────────────
if [ ! -f dataset/splits/10h.json ]; then
    echo "Creating labeled split manifests …"
    python dataset/splits.py --data_root ./dataset --seed 42
fi

# ── Pre-train ──────────────────────────────────────────────────────────────
# NOTE: HuBERT k-means is fit on rank-0 and cached to disk before DDP starts.
# If the cache already exists at outputs/hubert_kmeans/kmeans_k100.pkl, it
# will be loaded on all ranks without refitting.
torchrun \
    --nproc_per_node=8 \
    --master_port=29501 \
    baselines/pretrain_hubert.py \
        --data_root              ./dataset \
        --output_dir             ./outputs/hubert_pretrained \
        --hubert_kmeans_cache    ./outputs/hubert_kmeans \
        --hubert_kmeans_clusters 100 \
        --max_steps              400000 \
        --batch_size             32 \
        --lr                     5e-4 \
        --warmup_steps           32000 \
        --precision              bf16 \
        --save_steps             10000 \
        --eval_steps             10000 \
        --log_steps              200 \
        --num_workers            4 \
        --seed                   42
