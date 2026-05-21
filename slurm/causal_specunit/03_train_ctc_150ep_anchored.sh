#!/bin/bash
# Anchored fine-tune recipe: full 960h SSL CTC fine-tuning with four
# improvements over 03_train_ctc_150ep_fair_ssl_lpft_interctc.sh, each
# addressing a specific failure mode we observed in the current run.
#
# Changes from previous LP-FT+InterCTC recipe:
#
#   (1) Drop the 2-epoch encoder freeze, keep a 5-epoch encoder rewarmup
#       (FREEZE=0, REWARMUP=5). The freeze was costing us 2 epochs we never
#       recovered (ep 13 WER 35.77% vs old SSL 32.70%). The rewarmup alone
#       still gives a gentle ramp-up while letting the encoder receive
#       gradient from step 0.
#
#   (2) Lower InterCTC weight (0.30 → 0.15). At 0.30 the main CTC head was
#       capped at 70% of the loss budget. The InterCTC ratio (1.42 at ep 13)
#       is healthy, so we don't need that much weight to keep the encoder
#       receiving deep gradient. Giving more signal back to the main head
#       lowers WER directly.
#
#   (3) SpecAug uses the SSL mask_emb token (--specaug-mask-source ssl-mask)
#       instead of zero-fill. The encoder was pretrained to expect mask_emb
#       at corrupted frames. Zero-fill is OOD for the encoder; matching the
#       distribution removes that mismatch.
#
#   (4) SSL-target anchored fine-tuning (ssl_anchor_weight=0.1). The model
#       additionally predicts the K=100/K=500 SSL cluster IDs at every
#       encoder output position via two small auxiliary heads (warm-started
#       from the SSL pretraining heads). This anchors the encoder to its
#       pretrained feature space — prevents the fresh CTC head from
#       rewriting useful features during early adaptation.
#
# All four are SSL-specific or SSL-aware:
#   #1, #2 are recipe tunings (also valid for scratch).
#   #3 only makes sense if the encoder saw mask_emb during pretraining.
#   #4 is purely SSL — uses the precomputed cluster targets the user has
#       on disk; scratch has no targets to anchor against.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ctc960_anchored
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc960_anchored.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc960_anchored.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="${DATA_ROOT:-dataset/datasets/librispeech/LibriSpeech}"
TARGETS_DIR="${TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-dataset/bpe128.model}"
SSL_CHECKPOINT="${SSL_CHECKPOINT:-outputs/causal_specunit/pretrain_ssl_v2_150k_c8/checkpoint_step150000/checkpoint.pt}"

TRAIN_SPLITS="${TRAIN_SPLITS:-train-clean-100 train-clean-360 train-other-500}"
EVAL_SPLIT="${EVAL_SPLIT:-dev-other}"
EPOCHS="${EPOCHS:-150}"

# Same effective batch as the previous run: 128 * 4 * 1 = 512
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
WORKERS="${WORKERS:-8}"
DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-120}"

# Layer-wise LR (unchanged from the LP-FT recipe)
ENCODER_LR="${ENCODER_LR:-6e-4}"
HEAD_LR="${HEAD_LR:-1e-3}"
BASE_LR="${BASE_LR:-1e-3}"
ENCODER_LAYER_LR_DECAY="${ENCODER_LAYER_LR_DECAY:-0.85}"

# (1) DROP the 2-epoch freeze; keep a 5-epoch encoder rewarmup.
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
PEAK_EPOCHS="${PEAK_EPOCHS:-90}"
NOAM_DECAY_RATE="${NOAM_DECAY_RATE:-0.25}"
FREEZE_ENCODER_EPOCHS="${FREEZE_ENCODER_EPOCHS:-0}"
ENCODER_REWARMUP_EPOCHS="${ENCODER_REWARMUP_EPOCHS:-5}"

# (2) Lower InterCTC weight from 0.30 → 0.15.
INTER_CTC_LAYERS="${INTER_CTC_LAYERS:-7}"
INTER_CTC_WEIGHT="${INTER_CTC_WEIGHT:-0.15}"

# (3) SpecAug uses the SSL learned mask_emb token instead of zero-fill.
SPECAUG_MASK_SOURCE="${SPECAUG_MASK_SOURCE:-ssl-mask}"

# (4) SSL-target anchored fine-tuning. Auxiliary cluster prediction loss
# (K=100 + K=500) at weight 0.1 alongside CTC, with anchor heads warm-started
# from the SSL pretraining heads.
SSL_ANCHOR_WEIGHT="${SSL_ANCHOR_WEIGHT:-0.1}"
SSL_ANCHOR_TARGETS_DIR="${SSL_ANCHOR_TARGETS_DIR:-${TARGETS_DIR}}"
SSL_ANCHOR_LOAD_HEADS="${SSL_ANCHOR_LOAD_HEADS:-1}"

MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NO_DECAY_NORM_BIAS="${NO_DECAY_NORM_BIAS:-1}"

SPECAUG_TIME_MASK_PARAM="${SPECAUG_TIME_MASK_PARAM:-40}"
SPECAUG_FREQ_MASK_PARAM="${SPECAUG_FREQ_MASK_PARAM:-30}"
SPECAUG_TIME_MASKS="${SPECAUG_TIME_MASKS:-2}"
SPECAUG_FREQ_MASKS="${SPECAUG_FREQ_MASKS:-2}"
SPECAUG_DISABLE_LAST_EPOCHS="${SPECAUG_DISABLE_LAST_EPOCHS:-15}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/ctc_ssl_anchored_960h_v2}"

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

[ -d "${VIRTUAL_ENV}" ] || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}" ]   || { echo "Missing data root: ${DATA_ROOT}"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]    || { echo "Missing CMVN"; exit 1; }
[ -f "${TARGETS_DIR}/targets.pt" ] || { echo "Missing SSL targets for the anchor objective"; exit 1; }
[ -f "${TOKENIZER_PATH}" ] || { echo "Missing tokenizer"; exit 1; }
[ -f "${SSL_CHECKPOINT}" ] || { echo "Missing SSL checkpoint: ${SSL_CHECKPOINT}"; exit 1; }

read -r -a TRAIN_SPLIT_ARGS <<< "${TRAIN_SPLITS}"
read -r -a INTER_CTC_LAYERS_ARGS <<< "${INTER_CTC_LAYERS}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((29000 + SLURM_JOB_ID % 20000))}"

NUM_PROCESSES=4

echo "Job ${SLURM_JOB_ID} anchored fine-tune starting at $(date)"
echo "Output:               ${OUTPUT_DIR}"
echo "SSL checkpoint:       ${SSL_CHECKPOINT}"
echo "Train splits:         ${TRAIN_SPLITS}"
echo "Eval split:           ${EVAL_SPLIT}"
echo "Epochs:               ${EPOCHS}"
echo "Effective batch:      $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
echo "LP-FT:                freeze=${FREEZE_ENCODER_EPOCHS}ep rewarmup=${ENCODER_REWARMUP_EPOCHS}ep"
echo "InterCTC:             layers=[${INTER_CTC_LAYERS}] weight=${INTER_CTC_WEIGHT}"
echo "SpecAug:              mask_source=${SPECAUG_MASK_SOURCE} time=${SPECAUG_TIME_MASK_PARAM}x${SPECAUG_TIME_MASKS} freq=${SPECAUG_FREQ_MASK_PARAM}x${SPECAUG_FREQ_MASKS} disable_last=${SPECAUG_DISABLE_LAST_EPOCHS}"
echo "SSL anchor:           weight=${SSL_ANCHOR_WEIGHT} targets=${SSL_ANCHOR_TARGETS_DIR} load_heads=${SSL_ANCHOR_LOAD_HEADS}"
echo "LR groups:            encoder_top=${ENCODER_LR} layer_decay=${ENCODER_LAYER_LR_DECAY} head=${HEAD_LR}"
echo "Schedule:             warmup=${WARMUP_EPOCHS} hold=${PEAK_EPOCHS} decay=${NOAM_DECAY_RATE}"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY

EXTRA_ARGS=(
    --encoder-layer-lr-decay "${ENCODER_LAYER_LR_DECAY}"
    --specaug-mask-source "${SPECAUG_MASK_SOURCE}"
    --freeze-encoder-epochs "${FREEZE_ENCODER_EPOCHS}"
    --encoder-rewarmup-epochs "${ENCODER_REWARMUP_EPOCHS}"
    --inter-ctc-layers "${INTER_CTC_LAYERS_ARGS[@]}"
    --inter-ctc-weight "${INTER_CTC_WEIGHT}"
    --ssl-anchor-weight "${SSL_ANCHOR_WEIGHT}"
    --ssl-anchor-targets-dir "${SSL_ANCHOR_TARGETS_DIR}"
)
if [ "${NO_DECAY_NORM_BIAS}" = "1" ]; then
    EXTRA_ARGS+=(--no-decay-norm-and-bias)
fi
if [ "${SSL_ANCHOR_LOAD_HEADS}" = "1" ]; then
    EXTRA_ARGS+=(--ssl-anchor-load-heads)
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
    "${EXTRA_ARGS[@]}" \
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

echo "Job ${SLURM_JOB_ID} anchored fine-tune finished at $(date)"
echo "Metrics: ${OUTPUT_DIR}/ctc_metrics.jsonl"
