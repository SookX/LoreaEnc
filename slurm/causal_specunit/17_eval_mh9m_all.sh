#!/bin/bash
# Evaluate every MelHuBERT-Transformer fine-tune cell on test-clean and
# test-other, writing eval_results.json next to each cell's
# checkpoint_best/checkpoint.pt. Skips cells whose eval_results.json is
# already present unless FORCE_EVAL=1.
#
# Submit:
#   sbatch slurm/causal_specunit/17_eval_mh9m_all.sh
#
# Optional filters:
#   SUBSETS="librilight_10h" sbatch ...
#   SEED_LIST="42 43" sbatch ...
#   FORCE_EVAL=1 sbatch ...

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=mh9m_eval
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/mh9m_eval.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/mh9m_eval.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-dataset/bpe128.model}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/causal_specunit/mh9m_ft}"
VARIANT="${VARIANT:-mh9m}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"

if [ -n "${SUBSETS:-}" ]; then
    read -r -a DATASETS <<< "${SUBSETS}"
else
    DATASETS=(librilight_1h librilight_10h train-clean-100)
fi

if [ -n "${SEED_LIST:-}" ]; then
    read -r -a SEEDS <<< "${SEED_LIST}"
else
    SEEDS=(42 43 44)
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
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
mkdir -p logs

[ -d "${VIRTUAL_ENV}" ]            || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}/test-clean" ]   || { echo "Missing split: ${DATA_ROOT}/test-clean"; exit 1; }
[ -d "${DATA_ROOT}/test-other" ]   || { echo "Missing split: ${DATA_ROOT}/test-other"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]    || { echo "Missing CMVN: ${TARGETS_DIR}/cmvn.pt"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]         || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
PORT_BASE=$((34000 + (SLURM_JOB_ID % 2500)))
CELL_IDX=0

log_phase() {
    echo ""
    echo "===================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "===================================================="
}

log_phase "EVAL ${VARIANT} subsets=${DATASETS[*]} seeds=${SEEDS[*]} force=${FORCE_EVAL:-0}"

for SUBSET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CELL_DIR="${OUTPUT_ROOT}/${SUBSET}/seed${SEED}"
        CKPT="${CELL_DIR}/checkpoint_best/checkpoint.pt"
        RESULT="${CELL_DIR}/eval_results.json"

        log_phase "CELL ${CELL_IDX}: ${SUBSET} seed=${SEED}"

        if [ ! -f "${CKPT}" ]; then
            echo "  SKIP: missing checkpoint ${CKPT}"
            CELL_IDX=$((CELL_IDX + 1))
            continue
        fi
        if [ -f "${RESULT}" ] && [ "${FORCE_EVAL:-0}" != "1" ]; then
            echo "  SKIP: eval_results.json already exists (set FORCE_EVAL=1 to re-run)"
            CELL_IDX=$((CELL_IDX + 1))
            continue
        fi

        EVAL_PORT=$((PORT_BASE + CELL_IDX))

        torchrun \
            --nproc_per_node="${NPROC_PER_NODE}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${EVAL_PORT}" \
            -m CausalSpecUnit.evaluate_ctc \
            --checkpoint "${CKPT}" \
            --data-root "${DATA_ROOT}" \
            --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
            --tokenizer-path "${TOKENIZER_PATH}" \
            --variant "${VARIANT}" \
            --splits test-clean test-other \
            --batch-size "${EVAL_BATCH_SIZE}" \
            --workers 4 \
            --output "${RESULT}"

        CELL_IDX=$((CELL_IDX + 1))
    done
done

log_phase "DONE — evaluated ${CELL_IDX} cells. Aggregate with: python scripts/aggregate_mh9m.py"
