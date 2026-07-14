#!/bin/bash
# Scratch 960h fair-comparison run. Identical recipe to the SSL 960h fine-tune
# (encoder_lr 3e-4, head_lr 1e-3, warmup 10, peak 50, decay 0.5, SpecAug, etc.)
# Only difference: no --ssl-checkpoint, so the encoder starts from random init.
# Trains 150 epochs, then evaluates on test-clean and test-other.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_test_scratch_960h
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_test_scratch_960h.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_test_scratch_960h.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
TOKENIZER_PATH="dataset/bpe128.model"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/test_table/scratch_960h}"

# Recipe — IDENTICAL to the SSL 960h fine-tune. The only delta is no --ssl-checkpoint.
# Effective batch = 128 * 4 * 1 = 512 (matches the SSL 960h run's 128*2*2 = 512;
# per-GPU batch and total gradient noise scale are unchanged across runs).
EPOCHS=150
BATCH_SIZE=128
GRAD_ACCUM_STEPS=1
EVAL_BATCH_SIZE=128
WORKERS=8
DATALOADER_TIMEOUT=120

ENCODER_LR=3e-4
HEAD_LR=1e-3
BASE_LR=1e-3
WARMUP_EPOCHS=10
PEAK_EPOCHS=50
NOAM_DECAY_RATE=0.5
MAX_GRAD_NORM=1.0

SPECAUG_TIME_MASK_PARAM=40
SPECAUG_FREQ_MASK_PARAM=30
SPECAUG_TIME_MASKS=2
SPECAUG_FREQ_MASKS=2
SPECAUG_DISABLE_LAST_EPOCHS=15

TRAIN_SPLITS="train-clean-100 train-clean-360 train-other-500"
EVAL_SPLIT=dev-other

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

[ -d "${VIRTUAL_ENV}" ]         || { echo "Missing venv"; exit 1; }
[ -d "${DATA_ROOT}" ]           || { echo "Missing data root"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ] || { echo "Missing CMVN"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]      || { echo "Missing tokenizer"; exit 1; }

read -r -a TRAIN_SPLIT_ARGS <<< "${TRAIN_SPLITS}"

PORT_BASE=$((24000 + SLURM_JOB_ID % 10000))
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# Final result lives at ${OUTPUT_DIR}/eval_results.json. If it exists, the
# whole job is a no-op — useful if you need to resubmit after a node failure.
if [ -f "${OUTPUT_DIR}/eval_results.json" ]; then
    echo "SKIP: ${OUTPUT_DIR}/eval_results.json already exists"
    exit 0
fi

# ---- Train (skip if checkpoint_best is already there) ----
if [ ! -f "${OUTPUT_DIR}/checkpoint_best/checkpoint.pt" ]; then
    export MASTER_PORT=$((PORT_BASE + 1))
    torchrun \
        --nproc_per_node=4 \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m CausalSpecUnit.train_ctc \
        --data-root "${DATA_ROOT}" \
        --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --train-splits "${TRAIN_SPLIT_ARGS[@]}" \
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
fi

# ---- Eval on test-clean + test-other ----
export MASTER_PORT=$((PORT_BASE + 2))
torchrun \
    --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m CausalSpecUnit.evaluate_ctc \
    --checkpoint "${OUTPUT_DIR}/checkpoint_best/checkpoint.pt" \
    --data-root "${DATA_ROOT}" \
    --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --variant xs \
    --splits test-clean test-other \
    --batch-size 64 \
    --workers 4 \
    --output "${OUTPUT_DIR}/eval_results.json"

echo "DONE scratch_960h: ${OUTPUT_DIR}/eval_results.json"
