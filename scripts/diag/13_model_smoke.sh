#!/bin/bash
# Standalone smoke test for the MelHuBERT-Transformer CTC pipeline.
# 1 GPU, synthetic data, no DDP, no dataloader. Confirms the model and
# the SSL checkpoint load are healthy. Finishes in <2 minutes.
#
# Submit:
#   sbatch scripts/diag/13_model_smoke.sh

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=diag_model
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/diag_model.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/diag_model.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"

SSL_CKPT="${SSL_CKPT:-outputs/causal_specunit/melhubert_transformer_mh9m/ssl_fine_150000/checkpoint_step150000/checkpoint.pt}"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"

python scripts/diag/13_model_smoke.py \
    --ssl-checkpoint "${SSL_CKPT}" \
    --variant mh9m \
    --batch-size 32 \
    --seq-len 1600 \
    --steps 10
