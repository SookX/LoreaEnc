#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=sqformer_xs
#SBATCH --time=72:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:8

#SBATCH -o logs/sqformer_xs.%j.out
#SBATCH -e logs/sqformer_xs.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

export VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
if [ ! -d "${VIRTUAL_ENV}" ]; then
    echo "Missing venv: ${VIRTUAL_ENV}"
    exit 1
fi
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
cd "${PROJECT_DIR}"

mkdir -p logs outputs/squeezeformer_xs_150ep

TOKENIZER_PATH="dataset/bpe128.model"
if [ ! -f "${TOKENIZER_PATH}" ]; then
    echo "Missing tokenizer: ${TOKENIZER_PATH}"
    exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-12355}"
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

NUM_PROCESSES="${SLURM_GPUS_ON_NODE:-8}"

echo "Job ${SLURM_JOB_ID} starting at $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Python: $(which python)"
echo "Accelerate: $(which accelerate)"
echo "GPUs on node: ${NUM_PROCESSES}"

python - <<'PY'
import torch
import accelerate
print("PyTorch:", torch.__version__)
print("Accelerate:", accelerate.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY

accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    SqueezeFormer/train.py \
    --epochs 150 \
    --variant xs \
    --eval-split dev-other \
    --eval-every 1 \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --batch-size 128 \
    --eval-batch-size 128 \
    --workers 8 \
    --output-dir outputs/squeezeformer_xs_150ep \
    --run-name squeezeformer_xs_150ep

echo "Job ${SLURM_JOB_ID} finished at $(date)"
