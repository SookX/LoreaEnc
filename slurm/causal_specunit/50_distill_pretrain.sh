#!/bin/bash
# Knowledge-distillation pretraining baseline (reviewer-requested).
#
# Distills a frozen, released 95M teacher (HuBERT Base or wav2vec2 Base) into
# the SAME compact 9M SqueezeFormer-XS encoder used everywhere else in the
# paper. This is the "competitive low-footprint alternative" the reviewer
# asked for: identical student backbone and inference cost as our SSL rows,
# only the pretraining strategy differs (DistilHuBERT-style feature distill
# vs. our spectrogram SSL).
#
# Two runs give two table rows:
#   TEACHER=hubert_base   sbatch slurm/causal_specunit/50_distill_pretrain.sh
#   TEACHER=wav2vec2_base sbatch slurm/causal_specunit/50_distill_pretrain.sh
#
# The distilled encoder saves under encoder.* keys, so fine-tuning is the
# UNCHANGED train_ctc.py — point any of the finetune scripts at it:
#   SSL_CKPT=outputs/causal_specunit/distill_hubert_base_960h/checkpoint_step250000/checkpoint.pt \
#   sbatch slurm/causal_specunit/41_finetune_m95_1h_seed42.sh   # (with --variant xs)
#
# Optional knobs:
#   TEACHER=wav2vec2_base MAX_STEPS=150000 BATCH_SIZE=48 sbatch ...

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_distill
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_distill.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_distill.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
# CMVN must match what train_ctc uses at fine-tune time so the student sees the
# same normalized mels in both stages. The finetune scripts use targets_960h_c8.
SOURCE_TARGETS_DIR="${SOURCE_TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"

# Teacher: hubert_base (default) or wav2vec2_base. DistilHuBERT predicts HuBERT
# layers 4/8/12 == 0-indexed 3/7/11.
TEACHER="${TEACHER:-hubert_base}"
TEACHER_LAYERS="${TEACHER_LAYERS:-3 7 11}"

# RANDOM_TEACHER=1 runs the random-init-teacher ablation: same architecture,
# same recipe, teacher weights carry NO pretraining. Distinct output dir so it
# never clobbers the real KD run. Propagates across the autochain via --export=ALL.
RANDOM_TEACHER="${RANDOM_TEACHER:-0}"
if [ "${RANDOM_TEACHER}" = "1" ]; then
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/distill_${TEACHER}_random_960h}"
    RANDOM_TEACHER_ARG="--random-teacher"
else
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/distill_${TEACHER}_960h}"
    RANDOM_TEACHER_ARG=""
fi
DURATIONS_CACHE="${DURATIONS_CACHE:-outputs/causal_specunit/librispeech_durations.json}"

# Cache the teacher checkpoint inside the project so it persists across jobs.
export TORCH_HOME="${TORCH_HOME:-${PROJECT_DIR}/.torch_cache}"

# Matches the paper's TOTAL neural budget: iter-1 (150k) + iter-2 (100k) = 250k.
# Distillation is single-stage, so it runs all 250k in one go. NB: a distill step
# runs a frozen 95M teacher forward, so 250k distill steps cost MORE GPU-hours than
# our 9M SSL — report matched-steps + measured GPU-h, don't claim a GPU-hour tie.
MAX_STEPS="${MAX_STEPS:-250000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-1e-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
PEAK_EPOCHS="${PEAK_EPOCHS:-90}"
MAX_DURATION_SEC="${MAX_DURATION_SEC:-20}"
# Optional: reuse an existing CMVN mel cache. Leave EMPTY unless the cache was
# baked with the SAME cmvn as SOURCE_TARGETS_DIR — otherwise the student would
# be distilled on differently-normalized mels than it is fine-tuned on.
MEL_CACHE_DIR="${MEL_CACHE_DIR:-}"

# Sanity checks
[ -d "${VIRTUAL_ENV}" ]                 || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}" ]                   || { echo "Missing data root: ${DATA_ROOT}"; exit 1; }
[ -f "${SOURCE_TARGETS_DIR}/cmvn.pt" ]  || { echo "Missing CMVN: ${SOURCE_TARGETS_DIR}/cmvn.pt"; exit 1; }
case "${TEACHER}" in
    hubert_base|wav2vec2_base) ;;
    *) echo "TEACHER must be hubert_base or wav2vec2_base, got: ${TEACHER}"; exit 1 ;;
esac

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((44000 + SLURM_JOB_ID % 20000))}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
mkdir -p logs "${OUTPUT_DIR}" "${TORCH_HOME}"

NUM_PROCESSES=2
WORKERS=12
DATALOADER_TIMEOUT=300

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] KD pretrain: ${TEACHER} -> SqueezeFormer-XS (9M)"
echo "  teacher=${TEACHER}  layers=[${TEACHER_LAYERS}]"
echo "  max_steps=${MAX_STEPS} batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM_STEPS} lr=${LR}"
echo "  cmvn=${SOURCE_TARGETS_DIR}/cmvn.pt"
echo "  max_duration=${MAX_DURATION_SEC}s  mel_cache='${MEL_CACHE_DIR:-<none, on-the-fly>}'"
echo "  effective batch = ${BATCH_SIZE} x ${NUM_PROCESSES} x ${GRAD_ACCUM_STEPS} = $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
echo "  output=${OUTPUT_DIR}"
echo "  TORCH_HOME=${TORCH_HOME}"
echo "===================================================="
echo ""

# Pre-flight: materialize the teacher checkpoint (downloads once into TORCH_HOME).
# Fails fast with guidance if the compute node has no internet.
python - "$TEACHER" <<'PY' || { echo "Teacher load failed. If compute nodes lack internet, pre-cache on a login node with the same TORCH_HOME, e.g.: TORCH_HOME=${TORCH_HOME} python -c \"import torchaudio; torchaudio.pipelines.HUBERT_BASE.get_model()\""; exit 1; }
import sys, torch, torchaudio
name = sys.argv[1]
pipe = {"hubert_base": "HUBERT_BASE", "wav2vec2_base": "WAV2VEC2_BASE"}[name]
m = getattr(torchaudio.pipelines, pipe).get_model()
n = sum(p.numel() for p in m.parameters()) / 1e6
print(f"Teacher {name} ({pipe}) ready: {n:.1f}M params")
print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
PY

T0=$(date +%s)

# 4h wall-time cap: resume from the latest checkpoint and queue a successor so
# the full run completes across a chain of 4h jobs. Exits here if MAX_STEPS is
# already reached. Sets RESUME_CKPT, consumed by the block just below.
SELF_SCRIPT="slurm/causal_specunit/50_distill_pretrain.sh"
source slurm/causal_specunit/_autochain.sh

RESUME_CKPT="${RESUME_CKPT:-}"
RESUME_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
    [ -f "${RESUME_CKPT}/checkpoint.pt" ] || [ -f "${RESUME_CKPT}" ] || { echo "RESUME_CKPT not found: ${RESUME_CKPT}"; exit 1; }
    echo "Resuming from checkpoint: ${RESUME_CKPT}"
    RESUME_ARGS=(--resume "${RESUME_CKPT}")
fi

MEL_CACHE_ARGS=()
if [ -n "${MEL_CACHE_DIR}" ]; then
    MEL_CACHE_ARGS=(--mel-cache-dir "${MEL_CACHE_DIR}")
fi

torchrun \
    --nproc_per_node="${NUM_PROCESSES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m CausalSpecUnit.distill_pretrain \
    --data-root "${DATA_ROOT}" \
    --cmvn-path "${SOURCE_TARGETS_DIR}/cmvn.pt" \
    --output-dir "${OUTPUT_DIR}" \
    --teacher "${TEACHER}" \
    --teacher-layers ${TEACHER_LAYERS} \
    ${RANDOM_TEACHER_ARG} \
    --variant xs \
    --epochs 1000 \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --lr "${LR}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --peak-epochs "${PEAK_EPOCHS}" \
    --noam-decay-rate 1.0 \
    --max-grad-norm 1.0 \
    --max-duration-sec "${MAX_DURATION_SEC}" \
    --bucket-sampler \
    --durations-cache "${DURATIONS_CACHE}" \
    --workers "${WORKERS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --prefetch-factor 4 \
    --log-every 50 \
    --save-every 1 \
    --keep-checkpoints 5 \
    --progress off \
    "${MEL_CACHE_ARGS[@]}" \
    "${RESUME_ARGS[@]}"

T1=$(date +%s)
ELAPSED=$((T1 - T0))

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] KD pretrain DONE (${TEACHER})"
echo "  elapsed: ${ELAPSED} s = $((ELAPSED / 3600)) h $((ELAPSED % 3600 / 60)) m"
echo "  steps:   ${MAX_STEPS}"
python3 -c "print(f'  throughput: {${ELAPSED} / ${MAX_STEPS} * 1000:.1f} ms/step')"
echo "===================================================="
echo "Distilled encoder: ${OUTPUT_DIR}/checkpoint_step$(printf '%06d' ${MAX_STEPS})/checkpoint.pt"
echo ""
echo "Fine-tune it (variant xs) via train_ctc / the finetune scripts, e.g.:"
echo "  SSL_CKPT=${OUTPUT_DIR}/checkpoint_step$(printf '%06d' ${MAX_STEPS})/checkpoint.pt"
