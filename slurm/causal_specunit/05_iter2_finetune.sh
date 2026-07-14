#!/bin/bash
# Simple CTC fine-tune from the iter-2 SSL checkpoint (4 GPUs).
# Recipe = the proven simple one: encoder_lr=3e-4 / head_lr=1e-3,
# Noam warmup=10 / peak=50 / decay=0.5, SpecAug with zero-fill.
# No InterCTC, no LP-FT, no SSL anchor, no layer-decay.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ctc_iter2_ft
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc_iter2_ft.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc_iter2_ft.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-dataset/bpe128.model}"

# Consumes the iter-2 SSL checkpoint produced by 04_iter2_pretrain_chain.sh
SSL_CHECKPOINT="${SSL_CHECKPOINT:-outputs/causal_specunit/pretrain_ssl_iter2_from_v2_c8/checkpoint_step100000/checkpoint.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/ctc_ssl_iter2_simple_960h}"

TRAIN_SPLITS="${TRAIN_SPLITS:-train-clean-100 train-clean-360 train-other-500}"
EVAL_SPLIT="${EVAL_SPLIT:-dev-other}"
EPOCHS="${EPOCHS:-150}"

# Effective batch = 128 * 4 * 1 = 512 (matches the proven simple FT recipe scale)
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
WORKERS="${WORKERS:-8}"
DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-120}"

# Simple recipe — same as 03_train_ctc_150ep_fair_ssl.sh defaults
ENCODER_LR="${ENCODER_LR:-3e-4}"
HEAD_LR="${HEAD_LR:-1e-3}"
BASE_LR="${BASE_LR:-1e-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
PEAK_EPOCHS="${PEAK_EPOCHS:-50}"
NOAM_DECAY_RATE="${NOAM_DECAY_RATE:-0.5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

SPECAUG_TIME_MASK_PARAM="${SPECAUG_TIME_MASK_PARAM:-40}"
SPECAUG_FREQ_MASK_PARAM="${SPECAUG_FREQ_MASK_PARAM:-30}"
SPECAUG_TIME_MASKS="${SPECAUG_TIME_MASKS:-2}"
SPECAUG_FREQ_MASKS="${SPECAUG_FREQ_MASKS:-2}"
SPECAUG_DISABLE_LAST_EPOCHS="${SPECAUG_DISABLE_LAST_EPOCHS:-15}"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

[ -d "${VIRTUAL_ENV}" ]            || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}" ]              || { echo "Missing data root: ${DATA_ROOT}"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]    || { echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]         || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }
[ -f "${SSL_CHECKPOINT}" ]         || { echo "Missing SSL checkpoint: ${SSL_CHECKPOINT}"; exit 1; }

read -r -a TRAIN_SPLIT_ARGS <<< "${TRAIN_SPLITS}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((22000 + SLURM_JOB_ID % 20000))}"

NUM_PROCESSES=4

echo "Job ${SLURM_JOB_ID} iter-2 simple CTC fine-tune starting at $(date)"
echo "SSL checkpoint:   ${SSL_CHECKPOINT}"
echo "Output:           ${OUTPUT_DIR}"
echo "Train splits:     ${TRAIN_SPLITS}"
echo "Eval split:       ${EVAL_SPLIT}"
echo "Epochs:           ${EPOCHS}"
echo "Effective batch:  $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
echo "Recipe:           simple — encoder_lr=${ENCODER_LR} head_lr=${HEAD_LR} warmup=${WARMUP_EPOCHS} peak=${PEAK_EPOCHS} decay=${NOAM_DECAY_RATE}"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY

torchrun \
    --nproc_per_node="${NUM_PROCESSES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m CausalSpecUnit.train_ctc \
    --data-root "${DATA_ROOT}" \
    --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --train-splits "${TRAIN_SPLIT_ARGS[@]}" \
    --ssl-checkpoint "${SSL_CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --variant xs \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --eval-split "${EVAL_SPLIT}" \
    --eval-every 1 \
    --workers "${WORKERS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --lr "${BASE_LR}" \
    --encoder-lr "${ENCODER_LR}" \
    --head-lr "${HEAD_LR}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --peak-epochs "${PEAK_EPOCHS}" \
    --noam-decay-rate "${NOAM_DECAY_RATE}" \
    --max-grad-norm "${MAX_GRAD_NORM}" \
    --specaug \
    --specaug-time-mask-param "${SPECAUG_TIME_MASK_PARAM}" \
    --specaug-freq-mask-param "${SPECAUG_FREQ_MASK_PARAM}" \
    --specaug-time-masks "${SPECAUG_TIME_MASKS}" \
    --specaug-freq-masks "${SPECAUG_FREQ_MASKS}" \
    --specaug-disable-last-epochs "${SPECAUG_DISABLE_LAST_EPOCHS}" \
    --progress off \
    --log-every 0 \
    --save-every 10

echo "Job ${SLURM_JOB_ID} iter-2 simple CTC fine-tune finished at $(date)"
echo "Metrics: ${OUTPUT_DIR}/ctc_metrics.jsonl"
