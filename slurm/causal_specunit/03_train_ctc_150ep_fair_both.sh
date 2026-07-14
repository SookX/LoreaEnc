#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ctc960_fair_both
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc960_fair_both.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc960_fair_both.%j.err

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

TRAIN_SPLITS="${TRAIN_SPLITS:-train-clean-100 train-clean-360 train-other-500}"
EVAL_SPLIT="${EVAL_SPLIT:-dev-other}"
EPOCHS="${EPOCHS:-150}"
RUN_SCRATCH="${RUN_SCRATCH:-1}"
RUN_SSL="${RUN_SSL:-1}"

BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
WORKERS="${WORKERS:-8}"
DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-120}"

# Full-data comparison. Scratch keeps the original fair recipe; SSL gets a
# slightly more adaptive fine-tune recipe because the 960h curve under-adapts
# rather than overfits.
ENCODER_LR="${ENCODER_LR:-3e-4}"
HEAD_LR="${HEAD_LR:-1e-3}"
BASE_LR="${BASE_LR:-1e-3}"
SSL_ENCODER_LR="${SSL_ENCODER_LR:-4e-4}"
SSL_HEAD_LR="${SSL_HEAD_LR:-1e-3}"
SSL_BASE_LR="${SSL_BASE_LR:-1e-3}"
SSL_ENCODER_LAYER_LR_DECAY="${SSL_ENCODER_LAYER_LR_DECAY:-1.0}"
SSL_WARMUP_EPOCHS="${SSL_WARMUP_EPOCHS:-10}"
SSL_PEAK_EPOCHS="${SSL_PEAK_EPOCHS:-90}"
SSL_NOAM_DECAY_RATE="${SSL_NOAM_DECAY_RATE:-0.25}"
SSL_SPECAUG_MASK_SOURCE="${SSL_SPECAUG_MASK_SOURCE:-zero}"
SSL_NO_DECAY_NORM_BIAS="${SSL_NO_DECAY_NORM_BIAS:-1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
PEAK_EPOCHS="${PEAK_EPOCHS:-50}"
NOAM_DECAY_RATE="${NOAM_DECAY_RATE:-0.5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

SPECAUG_TIME_MASK_PARAM="${SPECAUG_TIME_MASK_PARAM:-40}"
SPECAUG_FREQ_MASK_PARAM="${SPECAUG_FREQ_MASK_PARAM:-30}"
SPECAUG_TIME_MASKS="${SPECAUG_TIME_MASKS:-2}"
SPECAUG_FREQ_MASKS="${SPECAUG_FREQ_MASKS:-2}"
SPECAUG_DISABLE_LAST_EPOCHS="${SPECAUG_DISABLE_LAST_EPOCHS:-15}"

SCRATCH_OUTPUT_DIR="${SCRATCH_OUTPUT_DIR:-outputs/causal_specunit/ctc_scratch_960h_specaug_fair_elr3e4_hlr1e3_w10_p50_150ep_c8}"
SSL_OUTPUT_DIR="${SSL_OUTPUT_DIR:-outputs/causal_specunit/ctc_ssl_960h_specaug_tune_elr4e4_ld100_hlr1e3_w10_p90_d025_150ep_c8}"

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
mkdir -p logs "${SCRATCH_OUTPUT_DIR}" "${SSL_OUTPUT_DIR}"

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
    echo "Run slurm/causal_specunit/01_generate_targets.sh first."
    exit 1
fi
if [ ! -f "${TOKENIZER_PATH}" ]; then
    echo "Missing tokenizer: ${TOKENIZER_PATH}"
    exit 1
fi
if [ ! -f "${SSL_CHECKPOINT}" ]; then
    echo "Missing SSL checkpoint: ${SSL_CHECKPOINT}"
    echo "Run slurm/causal_specunit/02_pretrain_ssl_100k_c8.sh first."
    exit 1
fi

read -r -a TRAIN_SPLIT_ARGS <<< "${TRAIN_SPLITS}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((23000 + SLURM_JOB_ID % 20000))}"

NUM_PROCESSES=2

echo "Job ${SLURM_JOB_ID} fair 960h/150ep SpecAug CTC comparison starting at $(date)"
echo "Python: $(which python)"
echo "Torchrun: $(which torchrun)"
echo "Data root: ${DATA_ROOT}"
echo "Train splits: ${TRAIN_SPLITS}"
echo "Eval split: ${EVAL_SPLIT}"
echo "Epochs: ${EPOCHS}"
echo "Run scratch: ${RUN_SCRATCH}"
echo "Run SSL: ${RUN_SSL}"
echo "Effective batch: $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
echo "Scratch recipe: base=${BASE_LR} encoder=${ENCODER_LR} head=${HEAD_LR} warmup=${WARMUP_EPOCHS} hold=${PEAK_EPOCHS} decay=${NOAM_DECAY_RATE}"
echo "SSL recipe: base=${SSL_BASE_LR} encoder_top=${SSL_ENCODER_LR} layer_decay=${SSL_ENCODER_LAYER_LR_DECAY} head=${SSL_HEAD_LR} warmup=${SSL_WARMUP_EPOCHS} hold=${SSL_PEAK_EPOCHS} decay=${SSL_NOAM_DECAY_RATE} specaug_mask_source=${SSL_SPECAUG_MASK_SOURCE} no_decay_norm_bias=${SSL_NO_DECAY_NORM_BIAS}"
echo "SpecAug: time=${SPECAUG_TIME_MASK_PARAM}x${SPECAUG_TIME_MASKS} freq=${SPECAUG_FREQ_MASK_PARAM}x${SPECAUG_FREQ_MASKS} disable_last=${SPECAUG_DISABLE_LAST_EPOCHS}"
echo "Scratch output: ${SCRATCH_OUTPUT_DIR}"
echo "SSL output: ${SSL_OUTPUT_DIR}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY

COMMON_ARGS=(
    -m CausalSpecUnit.train_ctc
    --data-root "${DATA_ROOT}"
    --cmvn-path "${TARGETS_DIR}/cmvn.pt"
    --tokenizer-path "${TOKENIZER_PATH}"
    --train-splits "${TRAIN_SPLIT_ARGS[@]}"
    --variant xs
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --eval-split "${EVAL_SPLIT}"
    --eval-every 1
    --workers "${WORKERS}"
    --dataloader-timeout "${DATALOADER_TIMEOUT}"
    --lr "${BASE_LR}"
    --encoder-lr "${ENCODER_LR}"
    --head-lr "${HEAD_LR}"
    --warmup-epochs "${WARMUP_EPOCHS}"
    --peak-epochs "${PEAK_EPOCHS}"
    --noam-decay-rate "${NOAM_DECAY_RATE}"
    --max-grad-norm "${MAX_GRAD_NORM}"
    --specaug
    --specaug-time-mask-param "${SPECAUG_TIME_MASK_PARAM}"
    --specaug-freq-mask-param "${SPECAUG_FREQ_MASK_PARAM}"
    --specaug-time-masks "${SPECAUG_TIME_MASKS}"
    --specaug-freq-masks "${SPECAUG_FREQ_MASKS}"
    --specaug-disable-last-epochs "${SPECAUG_DISABLE_LAST_EPOCHS}"
    --progress off
    --log-every 0
    --save-every 10
)

SSL_EXTRA_ARGS=(
    --lr "${SSL_BASE_LR}"
    --encoder-lr "${SSL_ENCODER_LR}"
    --head-lr "${SSL_HEAD_LR}"
    --encoder-layer-lr-decay "${SSL_ENCODER_LAYER_LR_DECAY}"
    --warmup-epochs "${SSL_WARMUP_EPOCHS}"
    --peak-epochs "${SSL_PEAK_EPOCHS}"
    --noam-decay-rate "${SSL_NOAM_DECAY_RATE}"
    --specaug-mask-source "${SSL_SPECAUG_MASK_SOURCE}"
)
if [ "${SSL_NO_DECAY_NORM_BIAS}" = "1" ]; then
    SSL_EXTRA_ARGS+=(--no-decay-norm-and-bias)
fi

if [ "${RUN_SCRATCH}" = "1" ]; then
    echo "Starting fair scratch 960h/150ep SpecAug run at $(date)"
    torchrun \
        --nproc_per_node="${NUM_PROCESSES}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        "${COMMON_ARGS[@]}" \
        --output-dir "${SCRATCH_OUTPUT_DIR}"
fi

if [ "${RUN_SSL}" = "1" ]; then
    if [ "${RUN_SCRATCH}" = "1" ]; then
        export MASTER_PORT="$((MASTER_PORT + 1))"
    fi
    echo "Starting tuned SSL 960h/150ep SpecAug run at $(date)"
    torchrun \
        --nproc_per_node="${NUM_PROCESSES}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        "${COMMON_ARGS[@]}" \
        --ssl-checkpoint "${SSL_CHECKPOINT}" \
        --output-dir "${SSL_OUTPUT_DIR}" \
        "${SSL_EXTRA_ARGS[@]}"
fi

echo "Job ${SLURM_JOB_ID} fair 960h/150ep SpecAug CTC comparison finished at $(date)"
echo "Metrics:"
echo "  ${SCRATCH_OUTPUT_DIR}/ctc_metrics.jsonl"
echo "  ${SSL_OUTPUT_DIR}/ctc_metrics.jsonl"
