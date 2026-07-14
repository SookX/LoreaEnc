#!/bin/bash
# Proof-of-concept sweep: 10min + 1h, 3 initializations x 1 seed,
# all in one 24h slurm allocation on 4 GPUs.
#
# Default layout: 2 datasets x 3 initializations x 1 seed = 6 fine-tunes.
#   datasets:   librilight_10min, librilight_1h
#   conditions: scratch, iter1, iter2
#   seeds:      42
#
# Each (condition, seed) cell:
#   1) Trains with size-scaled hparams selected by SUBSET
#   2) Evals checkpoint_best on test-clean + test-other
#   3) Writes eval_results.json
#
# Idempotent: any cell whose eval_results.json already exists is skipped on resubmit.
#
# Optional:
#   SUBSETS="librilight_10min librilight_1h librilight_10h" sbatch ...
#   SEED_LIST="42 43 44" sbatch ...
#   CLEAN_FIRST=1 sbatch ...   # wipe each cell's output dir before training
#                              # (NFS-safe: ignores .nfs* busy-file handles)

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_asr_poc4
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_asr_poc4.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_asr_poc4.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-dataset/bpe128.model}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/causal_specunit/asr_poc_4gpu_iter2_tuned}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

if [ -n "${SUBSETS:-}" ]; then
    read -r -a DATASETS <<< "${SUBSETS}"
else
    DATASETS=(librilight_10min librilight_1h)
fi
CONDITIONS=(scratch iter1 iter2)
if [ -n "${SEED_LIST:-}" ]; then
    read -r -a SEEDS <<< "${SEED_LIST}"
else
    SEEDS=(42)
fi

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
mkdir -p logs

# Sanity
[ -d "${VIRTUAL_ENV}" ]          || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}/dev-other" ]  || { echo "Missing eval split: ${DATA_ROOT}/dev-other"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]  || { echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]       || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }
for DATASET in "${DATASETS[@]}"; do
    [ -d "${DATA_ROOT}/${DATASET}" ] || { echo "Missing prepared split: ${DATA_ROOT}/${DATASET}"; exit 1; }
done

PORT_BASE=$((30000 + (SLURM_JOB_ID % 5000)))
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

log_phase() {
    echo ""
    echo "===================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "===================================================="
}

resolve_ssl_ckpt() {
    case "$1" in
        scratch) echo "" ;;
        iter1)   echo "outputs/causal_specunit/pretrain_ssl_v2_150k_c8/checkpoint_step150000/checkpoint.pt" ;;
        iter2)   echo "outputs/causal_specunit/pretrain_ssl_iter2_from_v2_c8/checkpoint_step100000/checkpoint.pt" ;;
        *)       echo "ERROR"; return 1 ;;
    esac
}

configure_recipe() {
    FREEZE_ENCODER_EPOCHS="0"
    ENCODER_REWARMUP_EPOCHS="0"
    case "${SUBSET}" in
        librilight_10min)
            # Commit 34dbc86 10min recipe, translated from 2 GPUs to 4 GPUs.
            # Original: batch=8 on 2 GPUs -> effective batch 16.
            # Current:  batch=4 on 4 GPUs -> effective batch 16.
            # All optimizer/schedule/SpecAug settings match 34dbc86.
            EPOCHS="150"
            BATCH_SIZE="4"
            GRAD_ACCUM_STEPS="1"
            EVAL_EVERY="1"
            SAVE_EVERY="10"
            ENCODER_LR="3e-4"
            HEAD_LR="1e-3"
            WARMUP_EPOCHS="10"
            PEAK_EPOCHS="50"
            MAX_GRAD_NORM="1.0"
            SPECAUG_TIME_MASK_PARAM="30"
            SPECAUG_FREQ_MASK_PARAM="20"
            SPECAUG_TIME_MASKS="2"
            SPECAUG_FREQ_MASKS="2"
            SPECAUG_DISABLE_LAST_EPOCHS="10"
            ;;
        librilight_1h)
            # Same 06_test_table.sh recipe. Batch=8 on 4 GPUs gives effective
            # 32, still intentionally smaller than the 10h effective batch 128.
            EPOCHS="150"
            BATCH_SIZE="8"
            GRAD_ACCUM_STEPS="1"
            EVAL_EVERY="1"
            SAVE_EVERY="10"
            ENCODER_LR="3e-4"
            HEAD_LR="1e-3"
            WARMUP_EPOCHS="10"
            PEAK_EPOCHS="50"
            MAX_GRAD_NORM="1.0"
            SPECAUG_TIME_MASK_PARAM="30"
            SPECAUG_FREQ_MASK_PARAM="20"
            SPECAUG_TIME_MASKS="2"
            SPECAUG_FREQ_MASKS="2"
            SPECAUG_DISABLE_LAST_EPOCHS="10"
            ;;
        librilight_10h)
            EPOCHS="150"
            BATCH_SIZE="16"
            GRAD_ACCUM_STEPS="1"
            EVAL_EVERY="1"
            SAVE_EVERY="10"
            ENCODER_LR="2e-4"
            HEAD_LR="1e-3"
            WARMUP_EPOCHS="10"
            PEAK_EPOCHS="50"
            MAX_GRAD_NORM="1.0"
            SPECAUG_TIME_MASK_PARAM="30"
            SPECAUG_FREQ_MASK_PARAM="20"
            SPECAUG_TIME_MASKS="2"
            SPECAUG_FREQ_MASKS="2"
            SPECAUG_DISABLE_LAST_EPOCHS="10"
            ;;
        train-clean-100)
            EPOCHS="150"
            BATCH_SIZE="64"
            GRAD_ACCUM_STEPS="2"
            EVAL_EVERY="1"
            SAVE_EVERY="10"
            ENCODER_LR="3e-4"
            HEAD_LR="1e-3"
            WARMUP_EPOCHS="10"
            PEAK_EPOCHS="50"
            MAX_GRAD_NORM="1.0"
            SPECAUG_TIME_MASK_PARAM="40"
            SPECAUG_FREQ_MASK_PARAM="30"
            SPECAUG_TIME_MASKS="2"
            SPECAUG_FREQ_MASKS="2"
            SPECAUG_DISABLE_LAST_EPOCHS="15"
            ;;
        *)
            echo "Unsupported SUBSET=${SUBSET}; expected librilight_10min, librilight_1h, librilight_10h, or train-clean-100"
            exit 1
            ;;
    esac

    LR="${LR:-1e-3}"
    NOAM_DECAY_RATE="${NOAM_DECAY_RATE:-0.5}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
    EVAL_TEST_BATCH_SIZE="${EVAL_TEST_BATCH_SIZE:-64}"
}

apply_condition_overrides() {
    if [ "${SUBSET}" = "librilight_10min" ] && [ "${CONDITION}" = "iter2" ]; then
        # Iter-2 was still deletion-heavy around 0.96-0.98 WER with the exact
        # 34dbc86 150-epoch schedule on 4 GPUs. Keep the known-good LR family,
        # but give this tiny split more optimizer steps and lighter masking.
        EPOCHS="300"
        BATCH_SIZE="2"
        SAVE_EVERY="20"
        WARMUP_EPOCHS="20"
        PEAK_EPOCHS="80"
        SPECAUG_TIME_MASK_PARAM="10"
        SPECAUG_FREQ_MASK_PARAM="5"
        SPECAUG_TIME_MASKS="1"
        SPECAUG_FREQ_MASKS="1"
        SPECAUG_DISABLE_LAST_EPOCHS="30"
    fi
}

CELL_IDX=0
TOTAL_CELLS=$((${#DATASETS[@]} * ${#CONDITIONS[@]} * ${#SEEDS[@]}))
log_phase "START datasets=${DATASETS[*]} conditions=${CONDITIONS[*]} seeds=${SEEDS[*]} nproc=${NPROC_PER_NODE} total_cells=${TOTAL_CELLS}"

for SUBSET in "${DATASETS[@]}"; do
    for CONDITION in "${CONDITIONS[@]}"; do
        configure_recipe
        apply_condition_overrides
        log_phase "RECIPE ${SUBSET}/${CONDITION}: epochs=${EPOCHS} per_gpu_batch=${BATCH_SIZE} accum=${GRAD_ACCUM_STEPS} encoder_lr=${ENCODER_LR} head_lr=${HEAD_LR} warmup=${WARMUP_EPOCHS} peak=${PEAK_EPOCHS} max_grad=${MAX_GRAD_NORM} specaug=${SPECAUG_TIME_MASK_PARAM}/${SPECAUG_FREQ_MASK_PARAM}x${SPECAUG_TIME_MASKS}/${SPECAUG_FREQ_MASKS} disable_last=${SPECAUG_DISABLE_LAST_EPOCHS}"

        SSL_CHECKPOINT=$(resolve_ssl_ckpt "${CONDITION}")
        if [ -n "${SSL_CHECKPOINT}" ] && [ ! -f "${SSL_CHECKPOINT}" ]; then
            echo "Missing SSL checkpoint for ${CONDITION}: ${SSL_CHECKPOINT}"
            exit 1
        fi
        # Build the conditional --ssl-checkpoint arg as an array. Bash arrays
        # pass cleanly via expansion at the call site.
        SSL_ARGS=()
        if [ -n "${SSL_CHECKPOINT}" ]; then
            SSL_ARGS+=(--ssl-checkpoint "${SSL_CHECKPOINT}")
        fi

        for SEED in "${SEEDS[@]}"; do
            OUT_DIR="${OUTPUT_ROOT}/${SUBSET}/${CONDITION}_seed${SEED}"
            mkdir -p "${OUT_DIR}"

            log_phase "CELL ${CELL_IDX}/${TOTAL_CELLS}: ${SUBSET} | ${CONDITION} | seed=${SEED}"

            # CLEAN_FIRST=1 wipes the cell directory before retraining. Use
            # this when a previous run produced broken checkpoints/metrics
            # (e.g. the collapse-to-blanks failure on librilight_10min).
            # NFS-safe: '.nfs*' busy-file handles are silently skipped instead
            # of crashing the entire rm.
            if [ "${CLEAN_FIRST:-0}" = "1" ] && [ -d "${OUT_DIR}" ]; then
                log_phase "  CLEAN_FIRST=1 → wiping ${OUT_DIR}"
                find "${OUT_DIR}" -mindepth 1 ! -name ".nfs*" -delete 2>/dev/null || true
            fi

            if [ -f "${OUT_DIR}/eval_results.json" ]; then
                log_phase "SKIP — eval_results.json exists at ${OUT_DIR}"
                CELL_IDX=$((CELL_IDX + 1))
                continue
            fi

            TRAIN_PORT=$((PORT_BASE + CELL_IDX * 2 + 1))
            EVAL_PORT=$((PORT_BASE + CELL_IDX * 2 + 2))

            # ---- Train (skip if checkpoint_best already exists) ----
            if [ ! -f "${OUT_DIR}/checkpoint_best/checkpoint.pt" ]; then
                log_phase "  train ${CONDITION} seed=${SEED}"
                torchrun \
                    --nproc_per_node="${NPROC_PER_NODE}" \
                    --master_addr="${MASTER_ADDR}" \
                    --master_port="${TRAIN_PORT}" \
                    -m CausalSpecUnit.train_ctc \
                    --data-root "${DATA_ROOT}" \
                    --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
                    --tokenizer-path "${TOKENIZER_PATH}" \
                    --train-splits "${SUBSET}" \
                    ${SSL_ARGS[@]+"${SSL_ARGS[@]}"} \
                    --output-dir "${OUT_DIR}" \
                    --variant xs \
                    --epochs "${EPOCHS}" \
                    --batch-size "${BATCH_SIZE}" \
                    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
                    --eval-batch-size "${EVAL_BATCH_SIZE}" \
                    --eval-split dev-other \
                    --eval-every "${EVAL_EVERY}" \
                    --workers 8 \
                    --dataloader-timeout 120 \
                    --lr "${LR}" \
                    --encoder-lr "${ENCODER_LR}" \
                    --head-lr "${HEAD_LR}" \
                    --warmup-epochs "${WARMUP_EPOCHS}" \
                    --peak-epochs "${PEAK_EPOCHS}" \
                    --noam-decay-rate "${NOAM_DECAY_RATE}" \
                    --max-grad-norm "${MAX_GRAD_NORM}" \
                    --freeze-encoder-epochs "${FREEZE_ENCODER_EPOCHS}" \
                    --encoder-rewarmup-epochs "${ENCODER_REWARMUP_EPOCHS}" \
                    --specaug \
                    --specaug-time-mask-param "${SPECAUG_TIME_MASK_PARAM}" \
                    --specaug-freq-mask-param "${SPECAUG_FREQ_MASK_PARAM}" \
                    --specaug-time-masks "${SPECAUG_TIME_MASKS}" \
                    --specaug-freq-masks "${SPECAUG_FREQ_MASKS}" \
                    --specaug-disable-last-epochs "${SPECAUG_DISABLE_LAST_EPOCHS}" \
                    --seed "${SEED}" \
                    --progress off \
                    --log-every 0 \
                    --save-every "${SAVE_EVERY}"
            else
                log_phase "  SKIP train — checkpoint_best exists"
            fi

            # ---- Eval on test-clean + test-other ----
            log_phase "  eval ${CONDITION} seed=${SEED}"
            torchrun \
                --nproc_per_node="${NPROC_PER_NODE}" \
                --master_addr="${MASTER_ADDR}" \
                --master_port="${EVAL_PORT}" \
                -m CausalSpecUnit.evaluate_ctc \
                --checkpoint "${OUT_DIR}/checkpoint_best/checkpoint.pt" \
                --data-root "${DATA_ROOT}" \
                --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
                --tokenizer-path "${TOKENIZER_PATH}" \
                --variant xs \
                --splits test-clean test-other \
                --batch-size "${EVAL_TEST_BATCH_SIZE}" \
                --workers 4 \
                --output "${OUT_DIR}/eval_results.json"

            CELL_IDX=$((CELL_IDX + 1))
        done
    done
done

log_phase "DONE all ${TOTAL_CELLS} cells"
echo "Results:"
for SUBSET in "${DATASETS[@]}"; do
    for CONDITION in "${CONDITIONS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo "  ${OUTPUT_ROOT}/${SUBSET}/${CONDITION}_seed${SEED}/eval_results.json"
        done
    done
done
