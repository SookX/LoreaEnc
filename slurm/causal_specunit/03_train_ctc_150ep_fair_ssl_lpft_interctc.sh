#!/bin/bash
# Full 960h CTC fine-tune from SSL with the three structural changes that
# address the SSL under-adaptation observed at scale:
#
#   (1) InterCTC at layer 7 (post-time-reduction bottleneck):
#       Auxiliary CTC head halfway through the encoder. Old recipe relied on
#       gradient from a single CTC head at the top traveling back through 16
#       SqueezeFormer blocks + U-net recover. By epoch 25 the SSL train loss
#       (0.74) was already 14% worse than scratch (0.65) under identical LRs,
#       so the encoder isn't getting enough task signal. InterCTC injects a
#       direct CTC gradient at the bottleneck. Weight 0.3 follows the standard
#       Lee & Watanabe (2021) recipe — final = 0.7*main + 0.3*inter.
#
#   (2) LP-FT (Linear Probe then Fine-Tune): freeze encoder 2 epochs, then
#       linearly re-warmup encoder LR over 3 epochs (epochs 3-5) while the
#       head schedule is unchanged. Rationale: with a randomly-initialized
#       head at full LR=1e-3, the encoder is being pulled by gradients fit to
#       junk head features in the first few epochs — this is where the SSL
#       basin starts to leak. Letting the head settle on the (frozen) SSL
#       features first, then bringing the encoder in gently, preserves the
#       pretrained representation.
#
#   (3) Layer-wise LR decay 0.85 with top encoder LR 6e-4:
#       Bottom encoder layers (low-level acoustic features) transfer well from
#       SSL — keep them stable. Top encoder layers (which in SSL were tuned for
#       cluster-id prediction) need to be reshaped for CTC — give them more
#       headroom. With 16 layers: top=6e-4, layer 8 (mid)≈1.95e-4 (similar to
#       old 3e-4 baseline), bottom=6e-4*0.85^15 ≈ 5.2e-5 (~11x top-to-bottom
#       range). Previous attempts at 0.96 were too uniform (1.7x range, like
#       no decay); at 0.75 the bottom would be near-frozen (~1.1e-5).

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_ctc960_ssl_lpft_ic
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc960_ssl_lpft_ic.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_ctc960_ssl_lpft_ic.%j.err

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

# Effective batch = BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS = 128 * 4 * 1 = 512.
# This matches the prior 2-GPU run (128 * 2 * 2 = 512), so LR/SpecAug stay numerically
# comparable to the scratch and old-SSL baselines — only wall clock changes.
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
WORKERS="${WORKERS:-8}"
DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-120}"

# (3) Layer-wise LR — top encoder LR + layer-decay 0.85
# Top layer sees 6e-4; layer 8 (mid) sees 6e-4*0.85^7 ≈ 1.95e-4; bottom sees ~5.2e-5.
# Conservative vs 8e-4 — the previous 5e-4 attempt clipped 30%, this gives only the top layer a real LR bump.
ENCODER_LR="${ENCODER_LR:-6e-4}"
HEAD_LR="${HEAD_LR:-1e-3}"
BASE_LR="${BASE_LR:-1e-3}"
ENCODER_LAYER_LR_DECAY="${ENCODER_LAYER_LR_DECAY:-0.85}"

# (2) LP-FT — head schedule (warmup/peak/decay) is the SAME as the old recipe so
# the head trajectory is directly comparable. Encoder is held at LR=0 for 2 ep,
# then re-warmed from 0 → 8e-4 over the next 3 ep, then tracks the same peak/decay.
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
PEAK_EPOCHS="${PEAK_EPOCHS:-90}"
NOAM_DECAY_RATE="${NOAM_DECAY_RATE:-0.25}"
FREEZE_ENCODER_EPOCHS="${FREEZE_ENCODER_EPOCHS:-2}"
ENCODER_REWARMUP_EPOCHS="${ENCODER_REWARMUP_EPOCHS:-3}"

# (1) InterCTC — head at layer 7 (post-time-reduction, deep enough to carry
# meaningful features, shallow enough that we're not just duplicating the main head).
INTER_CTC_LAYERS="${INTER_CTC_LAYERS:-7}"
INTER_CTC_WEIGHT="${INTER_CTC_WEIGHT:-0.3}"

MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SPECAUG_MASK_SOURCE="${SPECAUG_MASK_SOURCE:-zero}"
NO_DECAY_NORM_BIAS="${NO_DECAY_NORM_BIAS:-1}"

SPECAUG_TIME_MASK_PARAM="${SPECAUG_TIME_MASK_PARAM:-40}"
SPECAUG_FREQ_MASK_PARAM="${SPECAUG_FREQ_MASK_PARAM:-30}"
SPECAUG_TIME_MASKS="${SPECAUG_TIME_MASKS:-2}"
SPECAUG_FREQ_MASKS="${SPECAUG_FREQ_MASKS:-2}"
SPECAUG_DISABLE_LAST_EPOCHS="${SPECAUG_DISABLE_LAST_EPOCHS:-15}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/ctc_ssl_960h_lpft_interctc_elr6e4_ld085_w10_p90_d025_150ep_c8}"

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
read -r -a INTER_CTC_LAYERS_ARGS <<< "${INTER_CTC_LAYERS}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((27000 + SLURM_JOB_ID % 20000))}"

NUM_PROCESSES=4

echo "Job ${SLURM_JOB_ID} SSL 960h/150ep CTC fine-tune w/ LP-FT + InterCTC starting at $(date)"
echo "Python: $(which python)"
echo "Torchrun: $(which torchrun)"
echo "Data root: ${DATA_ROOT}"
echo "Train splits: ${TRAIN_SPLITS}"
echo "Eval split: ${EVAL_SPLIT}"
echo "Epochs: ${EPOCHS}"
echo "Effective batch: $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
echo "LR groups: encoder_top=${ENCODER_LR} layer_decay=${ENCODER_LAYER_LR_DECAY} head=${HEAD_LR} no_decay_norm_bias=${NO_DECAY_NORM_BIAS}"
echo "LP-FT: freeze=${FREEZE_ENCODER_EPOCHS}ep rewarmup=${ENCODER_REWARMUP_EPOCHS}ep"
echo "InterCTC: layers=[${INTER_CTC_LAYERS}] weight=${INTER_CTC_WEIGHT}"
echo "SpecAug: time=${SPECAUG_TIME_MASK_PARAM}x${SPECAUG_TIME_MASKS} freq=${SPECAUG_FREQ_MASK_PARAM}x${SPECAUG_FREQ_MASKS} disable_last=${SPECAUG_DISABLE_LAST_EPOCHS}"
echo "Head schedule: warmup=${WARMUP_EPOCHS} hold=${PEAK_EPOCHS} decay=${NOAM_DECAY_RATE}"
echo "SSL checkpoint: ${SSL_CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

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
)
if [ "${NO_DECAY_NORM_BIAS}" = "1" ]; then
    EXTRA_ARGS+=(--no-decay-norm-and-bias)
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

echo "Job ${SLURM_JOB_ID} SSL 960h/150ep CTC fine-tune w/ LP-FT + InterCTC finished at $(date)"
echo "Metrics: ${OUTPUT_DIR}/ctc_metrics.jsonl"
