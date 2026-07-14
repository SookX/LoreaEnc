#!/bin/bash
# Fair 100h CTC comparison: scratch vs iter-1 SSL vs iter-2 SSL.
# All three runs share data, subset seed, CMVN, optimizer groups, schedule,
# SpecAugment, batch shape, and evaluation. Only encoder initialization differs.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ctc100h_3way
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc100h_3way.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc100h_3way.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
ITER2_TARGETS_DIR="${ITER2_TARGETS_DIR:-outputs/causal_specunit/targets_iter2_ssl100k_c8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-dataset/bpe128.model}"
ITER1_SSL_CHECKPOINT="${ITER1_SSL_CHECKPOINT:-outputs/causal_specunit/pretrain_ssl_100k_c8/checkpoint_step100000/checkpoint.pt}"
ITER2_SSL_CHECKPOINT="${ITER2_SSL_CHECKPOINT:-outputs/causal_specunit/pretrain_ssl_iter2_100k_c8/checkpoint_step100000/checkpoint.pt}"

TRAIN_SPLITS="${TRAIN_SPLITS:-train-clean-100}"
TRAIN_HOURS="${TRAIN_HOURS:-100}"
SUBSET_SEED="${SUBSET_SEED:-42}"
EVAL_SPLIT="${EVAL_SPLIT:-dev-other}"
EPOCHS="${EPOCHS:-100}"
VARIANT="${VARIANT:-xs}"

BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
WORKERS="${WORKERS:-8}"
DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-120}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

# Fair 100h recipe. Keep these identical for scratch, iter-1, and iter-2.
ENCODER_LR="${ENCODER_LR:-3e-4}"
HEAD_LR="${HEAD_LR:-1e-3}"
BASE_LR="${BASE_LR:-1e-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
PEAK_EPOCHS="${PEAK_EPOCHS:-50}"
NOAM_DECAY_RATE="${NOAM_DECAY_RATE:-0.5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# Matches the fair 100h baseline recipe. Override explicitly for stronger masks.
SPECAUG_TIME_MASK_PARAM="${SPECAUG_TIME_MASK_PARAM:-30}"
SPECAUG_FREQ_MASK_PARAM="${SPECAUG_FREQ_MASK_PARAM:-20}"
SPECAUG_TIME_MASKS="${SPECAUG_TIME_MASKS:-2}"
SPECAUG_FREQ_MASKS="${SPECAUG_FREQ_MASKS:-2}"
SPECAUG_DISABLE_LAST_EPOCHS="${SPECAUG_DISABLE_LAST_EPOCHS:-30}"

RUN_SCRATCH="${RUN_SCRATCH:-1}"
RUN_ITER1="${RUN_ITER1:-1}"
RUN_ITER2="${RUN_ITER2:-1}"
ALLOW_EXISTING_OUTPUTS="${ALLOW_EXISTING_OUTPUTS:-0}"

SCRATCH_OUTPUT_DIR="${SCRATCH_OUTPUT_DIR:-outputs/causal_specunit/ctc_scratch_100h_3way_specaug_fair_elr3e4_hlr1e3_w10_p50_100ep_c8}"
ITER1_OUTPUT_DIR="${ITER1_OUTPUT_DIR:-outputs/causal_specunit/ctc_ssl_iter1_100h_3way_specaug_fair_elr3e4_hlr1e3_w10_p50_100ep_c8}"
ITER2_OUTPUT_DIR="${ITER2_OUTPUT_DIR:-outputs/causal_specunit/ctc_ssl_iter2_100h_3way_specaug_fair_elr3e4_hlr1e3_w10_p50_100ep_c8}"

export VIRTUAL_ENV
export TARGETS_DIR
export ITER2_TARGETS_DIR
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
mkdir -p logs

if [ ! -d "${VIRTUAL_ENV}" ]; then
    echo "Missing venv: ${VIRTUAL_ENV}"
    exit 1
fi
if [ ! -d "${DATA_ROOT}" ]; then
    echo "Missing data root: ${DATA_ROOT}"
    exit 1
fi
if [ ! -f "${TARGETS_DIR}/cmvn.pt" ]; then
    echo "Missing shared CMVN: ${TARGETS_DIR}/cmvn.pt"
    echo "Run slurm/causal_specunit/01_generate_targets.sh first."
    exit 1
fi
if [ ! -f "${TOKENIZER_PATH}" ]; then
    echo "Missing tokenizer: ${TOKENIZER_PATH}"
    exit 1
fi
if [ "${RUN_ITER1}" = "1" ] && [ ! -f "${ITER1_SSL_CHECKPOINT}" ]; then
    echo "Missing iter-1 SSL checkpoint: ${ITER1_SSL_CHECKPOINT}"
    echo "Run slurm/causal_specunit/02_pretrain_ssl_100k_c8.sh first."
    exit 1
fi
if [ "${RUN_ITER2}" = "1" ] && [ ! -f "${ITER2_SSL_CHECKPOINT}" ]; then
    echo "Missing iter-2 SSL checkpoint: ${ITER2_SSL_CHECKPOINT}"
    echo "Run slurm/causal_specunit/02_pretrain_ssl_iter2_100k_c8.sh first."
    exit 1
fi
if [ "${RUN_ITER2}" = "1" ] && [ -f "${ITER2_TARGETS_DIR}/metadata.json" ]; then
    echo "Found iter-2 target metadata: ${ITER2_TARGETS_DIR}/metadata.json"
elif [ "${RUN_ITER2}" = "1" ]; then
    echo "Warning: missing iter-2 metadata: ${ITER2_TARGETS_DIR}/metadata.json"
fi

read -r -a TRAIN_SPLIT_ARGS <<< "${TRAIN_SPLITS}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((24000 + SLURM_JOB_ID % 20000))}"

run_ctc() {
    local label="$1"
    local output_dir="$2"
    local ssl_checkpoint="${3:-}"

    if [ "${ALLOW_EXISTING_OUTPUTS}" != "1" ] && [ -f "${output_dir}/ctc_metrics.jsonl" ]; then
        echo "Refusing to append to existing metrics: ${output_dir}/ctc_metrics.jsonl"
        echo "Use a new output dir, disable this run, or set ALLOW_EXISTING_OUTPUTS=1."
        exit 1
    fi

    mkdir -p "${output_dir}"

    echo "Starting ${label} 100h SpecAug run at $(date)"
    echo "  output=${output_dir}"
    if [ -n "${ssl_checkpoint}" ]; then
        echo "  ssl_checkpoint=${ssl_checkpoint}"
    else
        echo "  ssl_checkpoint=none"
    fi
    echo "  master=${MASTER_ADDR}:${MASTER_PORT}"

    local ssl_args=()
    if [ -n "${ssl_checkpoint}" ]; then
        ssl_args=(--ssl-checkpoint "${ssl_checkpoint}")
    fi

    torchrun \
        --nproc_per_node="${NUM_PROCESSES}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m CausalSpecUnit.train_ctc \
        --data-root "${DATA_ROOT}" \
        --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --train-splits "${TRAIN_SPLIT_ARGS[@]}" \
        "${ssl_args[@]}" \
        --output-dir "${output_dir}" \
        --variant "${VARIANT}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
        --eval-batch-size "${EVAL_BATCH_SIZE}" \
        --eval-split "${EVAL_SPLIT}" \
        --eval-every 1 \
        --workers "${WORKERS}" \
        --dataloader-timeout "${DATALOADER_TIMEOUT}" \
        --train-subset-hours "${TRAIN_HOURS}" \
        --train-subset-seed "${SUBSET_SEED}" \
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

    echo "Finished ${label} at $(date)"
    MASTER_PORT="$((MASTER_PORT + 1))"
    export MASTER_PORT
}

echo "Job ${SLURM_JOB_ID} fair 100h three-way CTC comparison starting at $(date)"
echo "Python: $(which python)"
echo "Torchrun: $(which torchrun)"
echo "Data root: ${DATA_ROOT}"
echo "Shared CMVN: ${TARGETS_DIR}/cmvn.pt"
echo "Train splits: ${TRAIN_SPLITS}"
echo "Train hours: ${TRAIN_HOURS}"
echo "Subset seed: ${SUBSET_SEED}"
echo "Eval split: ${EVAL_SPLIT}"
echo "Variant: ${VARIANT}"
echo "Epochs: ${EPOCHS}"
echo "Effective batch: $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
echo "LR groups: encoder=${ENCODER_LR} head=${HEAD_LR} base=${BASE_LR}"
echo "SpecAug: time=${SPECAUG_TIME_MASK_PARAM}x${SPECAUG_TIME_MASKS} freq=${SPECAUG_FREQ_MASK_PARAM}x${SPECAUG_FREQ_MASKS} disable_last=${SPECAUG_DISABLE_LAST_EPOCHS}"
echo "Schedule: warmup=${WARMUP_EPOCHS} hold=${PEAK_EPOCHS} decay=${NOAM_DECAY_RATE}"
echo "Run flags: scratch=${RUN_SCRATCH} iter1=${RUN_ITER1} iter2=${RUN_ITER2}"
echo "Allow existing outputs: ${ALLOW_EXISTING_OUTPUTS}"
echo "Scratch output: ${SCRATCH_OUTPUT_DIR}"
echo "Iter-1 output: ${ITER1_OUTPUT_DIR}"
echo "Iter-2 output: ${ITER2_OUTPUT_DIR}"

python - <<'PY'
import json
import os
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())

metadata_path = os.path.join(os.environ["ITER2_TARGETS_DIR"], "metadata.json")
if os.path.exists(metadata_path):
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    print("Iter-2 target metadata:", {
        "target_features": metadata.get("target_features"),
        "source_ssl_checkpoint": metadata.get("source_ssl_checkpoint"),
        "num_target_utterances": metadata.get("num_target_utterances"),
        "num_encoder_frames": metadata.get("num_encoder_frames"),
    })
PY

if [ "${RUN_SCRATCH}" = "1" ]; then
    run_ctc "scratch" "${SCRATCH_OUTPUT_DIR}"
fi
if [ "${RUN_ITER1}" = "1" ]; then
    run_ctc "iter-1 SSL" "${ITER1_OUTPUT_DIR}" "${ITER1_SSL_CHECKPOINT}"
fi
if [ "${RUN_ITER2}" = "1" ]; then
    run_ctc "iter-2 SSL" "${ITER2_OUTPUT_DIR}" "${ITER2_SSL_CHECKPOINT}"
fi

echo "Job ${SLURM_JOB_ID} fair 100h three-way CTC comparison finished at $(date)"
echo "Metrics:"
if [ "${RUN_SCRATCH}" = "1" ]; then
    echo "  scratch: ${SCRATCH_OUTPUT_DIR}/ctc_metrics.jsonl"
fi
if [ "${RUN_ITER1}" = "1" ]; then
    echo "  iter-1:  ${ITER1_OUTPUT_DIR}/ctc_metrics.jsonl"
fi
if [ "${RUN_ITER2}" = "1" ]; then
    echo "  iter-2:  ${ITER2_OUTPUT_DIR}/ctc_metrics.jsonl"
fi
