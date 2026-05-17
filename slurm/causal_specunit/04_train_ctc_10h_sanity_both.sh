#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ctc10h_both
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc10h_both.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc10h_both.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-dataset/bpe128.model}"
SSL_CHECKPOINT="${SSL_CHECKPOINT:-outputs/causal_specunit/pretrain_ssl_100k_c8/checkpoint_step100000/checkpoint.pt}"

# Default to clean speech for a stable sanity comparison. Override with:
#   TRAIN_SPLITS="train-other-500" sbatch slurm/causal_specunit/04_train_ctc_10h_sanity_both.sh
TRAIN_SPLITS="${TRAIN_SPLITS:-train-clean-100}"
TRAIN_HOURS="${TRAIN_HOURS:-10}"
SUBSET_SEED="${SUBSET_SEED:-42}"
EVAL_SPLIT="${EVAL_SPLIT:-dev-other}"
EPOCHS="${EPOCHS:-150}"

BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
WORKERS="${WORKERS:-6}"
DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-120}"

SSL_OUTPUT_DIR="${SSL_OUTPUT_DIR:-outputs/causal_specunit/ctc_ssl_10h_150ep_c8}"
SCRATCH_OUTPUT_DIR="${SCRATCH_OUTPUT_DIR:-outputs/causal_specunit/ctc_scratch_10h_150ep_c8}"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

cd "${PROJECT_DIR}"
mkdir -p logs "${SSL_OUTPUT_DIR}" "${SCRATCH_OUTPUT_DIR}"

if [ ! -d "${VIRTUAL_ENV}" ]; then
    echo "Missing venv: ${VIRTUAL_ENV}"
    exit 1
fi
if [ ! -d "${DATA_ROOT}" ]; then
    echo "Missing data root: ${DATA_ROOT}"
    exit 1
fi
if [ ! -f "${TARGETS_DIR}/cmvn.pt" ]; then
    echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"
    exit 1
fi
if [ ! -f "${TOKENIZER_PATH}" ]; then
    echo "Missing tokenizer: ${TOKENIZER_PATH}"
    exit 1
fi
if [ ! -f "${SSL_CHECKPOINT}" ]; then
    echo "Missing SSL checkpoint: ${SSL_CHECKPOINT}"
    exit 1
fi

read -r -a TRAIN_SPLIT_ARGS <<< "${TRAIN_SPLITS}"

echo "Job ${SLURM_JOB_ID} 10h CTC sanity starting at $(date)"
echo "Python: $(which python)"
echo "Data root: ${DATA_ROOT}"
echo "Train splits: ${TRAIN_SPLITS}"
echo "Train hours: ${TRAIN_HOURS}"
echo "Eval split: ${EVAL_SPLIT}"
echo "SSL output: ${SSL_OUTPUT_DIR}"
echo "Scratch output: ${SCRATCH_OUTPUT_DIR}"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY

COMMON_ARGS=(
    --data-root "${DATA_ROOT}"
    --cmvn-path "${TARGETS_DIR}/cmvn.pt"
    --tokenizer-path "${TOKENIZER_PATH}"
    --variant xs
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --eval-split "${EVAL_SPLIT}"
    --eval-every 1
    --workers "${WORKERS}"
    --dataloader-timeout "${DATALOADER_TIMEOUT}"
    --train-subset-hours "${TRAIN_HOURS}"
    --train-subset-seed "${SUBSET_SEED}"
    --train-splits "${TRAIN_SPLIT_ARGS[@]}"
    --progress off
    --log-every 0
    --save-every 10
)

echo "Starting SSL 10h sanity run at $(date)"
python -m CausalSpecUnit.train_ctc \
    "${COMMON_ARGS[@]}" \
    --ssl-checkpoint "${SSL_CHECKPOINT}" \
    --output-dir "${SSL_OUTPUT_DIR}" \
    --lr 1e-3 \
    --encoder-lr 3e-4 \
    --head-lr 1e-3

echo "Starting scratch 10h sanity run at $(date)"
python -m CausalSpecUnit.train_ctc \
    "${COMMON_ARGS[@]}" \
    --output-dir "${SCRATCH_OUTPUT_DIR}" \
    --lr 1e-3

echo "Job ${SLURM_JOB_ID} 10h CTC sanity finished at $(date)"
echo "Metrics:"
echo "  ${SSL_OUTPUT_DIR}/ctc_metrics.jsonl"
echo "  ${SCRATCH_OUTPUT_DIR}/ctc_metrics.jsonl"
