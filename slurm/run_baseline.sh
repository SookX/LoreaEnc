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
#SBATCH --gres=gpu:2
#SBATCH -o logs/sqformer_xs.%j.out
#SBATCH -e logs/sqformer_xs.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
TOKENIZER_PATH="dataset/bpe128.model"
OUTPUT_DIR="outputs/squeezeformer_xs_150ep"
RESUME_CHECKPOINT="${OUTPUT_DIR}/checkpoint_best"
START_EPOCH=93

if [ ! -d "${VIRTUAL_ENV}" ]; then
    echo "Missing venv: ${VIRTUAL_ENV}"
    exit 1
fi

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -f "${TOKENIZER_PATH}" ]; then
    echo "Missing tokenizer: ${TOKENIZER_PATH}"
    exit 1
fi

if [ ! -d "${RESUME_CHECKPOINT}" ]; then
    echo "Missing resume checkpoint: ${RESUME_CHECKPOINT}"
    exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-12355}"
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-40}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

# Match the Slurm request above. Keeping this explicit avoids accidentally
# launching 8 processes when Slurm reports no GPU count variable.
NUM_PROCESSES=2
WORKERS=4

ACCELERATE_LOG_DIR="logs/accelerate_${SLURM_JOB_ID}"
mkdir -p "${ACCELERATE_LOG_DIR}"

ACCELERATE_DEBUG_ARGS=()
if [ "${ENABLE_ACCELERATE_DEBUG:-0}" = "1" ]; then
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    ACCELERATE_DEBUG_ARGS=(--debug --tee 3 --log_dir "${ACCELERATE_LOG_DIR}")
fi

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
    "${ACCELERATE_DEBUG_ARGS[@]}" \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    SqueezeFormer/train.py \
    --data-root dataset/datasets/librispeech/LibriSpeech \
    --epochs 150 \
    --variant xs \
    --eval-split dev-other \
    --eval-every 1 \
    --no-compile \
    --resume "${RESUME_CHECKPOINT}" \
    --start-epoch "${START_EPOCH}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --batch-size 128 \
    --grad-accum-steps 2 \
    --max-grad-norm 1.0 \
    --max-safe-grad-norm 50.0 \
    --eval-batch-size 128 \
    --workers "${WORKERS}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name squeezeformer_xs_150ep

echo "Job ${SLURM_JOB_ID} finished at $(date)"
