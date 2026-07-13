#!/bin/bash
# Distillation SMOKE test — de-risks the full 250k KD run before you spend a
# 45-60 h allocation. Runs ~20 real steps on dev-clean, exercising the exact
# production path that the unit tests could only mock:
#   real 95M teacher forward -> extract_features -> 2x downsample -> 9M student
#   -> L1+cosine loss -> backward -> DDP allreduce -> checkpoint save.
# Then it loads the saved checkpoint into CausalSpecUnitCTC to confirm the
# encoder.* contract holds on REAL distilled weights (not mocked ones).
#
# Uses 2 GPUs by default so the real DDP + bucket-sampler topology is tested,
# not just a single-process path.
#
# Submit:
#   TEACHER=hubert_base   sbatch slurm/causal_specunit/50_distill_smoke.sh
#   TEACHER=wav2vec2_base sbatch slurm/causal_specunit/50_distill_smoke.sh
#
# Optional knobs:
#   NUM_PROCESSES=1 sbatch ...   # single-GPU, faster to schedule
#   MAX_STEPS=50 BATCH_SIZE=16 sbatch ...

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=distill_smk
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/distill_smk.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/distill_smk.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
SOURCE_TARGETS_DIR="${SOURCE_TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"

TEACHER="${TEACHER:-hubert_base}"
TEACHER_LAYERS="${TEACHER_LAYERS:-3 7 11}"
SMOKE_SPLIT="${SMOKE_SPLIT:-dev-clean}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal_specunit/distill_smoke_${TEACHER}}"
DURATIONS_CACHE="${DURATIONS_CACHE:-outputs/causal_specunit/distill_smoke_durations.json}"

export TORCH_HOME="${TORCH_HOME:-${PROJECT_DIR}/.torch_cache}"

MAX_STEPS="${MAX_STEPS:-20}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_DURATION_SEC="${MAX_DURATION_SEC:-20}"

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
export MASTER_PORT="${MASTER_PORT:-$((42000 + SLURM_JOB_ID % 20000))}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
# Fresh output each smoke so a stale checkpoint can't mask a regression.
rm -rf "${OUTPUT_DIR}"
mkdir -p logs "${OUTPUT_DIR}" "${TORCH_HOME}"

NUM_PROCESSES="${NUM_PROCESSES:-2}"
WORKERS="${WORKERS:-8}"
DATALOADER_TIMEOUT=300

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DISTILL SMOKE: ${TEACHER} -> SqueezeFormer-XS"
echo "  gpus=${NUM_PROCESSES}  split=${SMOKE_SPLIT}  steps=${MAX_STEPS}  batch=${BATCH_SIZE}"
echo "  cmvn=${SOURCE_TARGETS_DIR}/cmvn.pt  output=${OUTPUT_DIR}"
echo "===================================================="
echo ""

# Pre-flight: materialize the teacher (downloads once into TORCH_HOME).
python - "$TEACHER" <<'PY' || { echo "Teacher load failed. Pre-cache on a login node with the same TORCH_HOME."; exit 1; }
import sys, torch, torchaudio
name = sys.argv[1]
pipe = {"hubert_base": "HUBERT_BASE", "wav2vec2_base": "WAV2VEC2_BASE"}[name]
m = getattr(torchaudio.pipelines, pipe).get_model()
print(f"Teacher {name} ({pipe}) ready: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
print("CUDA:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
PY

T0=$(date +%s)

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
    --variant xs \
    --splits "${SMOKE_SPLIT}" \
    --epochs 1 \
    --max-steps "${MAX_STEPS}" \
    --max-train-batches "${MAX_TRAIN_BATCHES}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps 1 \
    --lr 1e-3 \
    --warmup-epochs 1 \
    --peak-epochs 1 \
    --max-grad-norm 1.0 \
    --max-duration-sec "${MAX_DURATION_SEC}" \
    --bucket-sampler \
    --durations-cache "${DURATIONS_CACHE}" \
    --workers "${WORKERS}" \
    --dataloader-timeout "${DATALOADER_TIMEOUT}" \
    --prefetch-factor 2 \
    --log-every 5 \
    --save-every 1 \
    --keep-checkpoints 2 \
    --progress on

T1=$(date +%s)
ELAPSED=$((T1 - T0))

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SMOKE training loop DONE in ${ELAPSED}s"
python3 -c "print(f'  throughput: {${ELAPSED} / ${MAX_STEPS} * 1000:.0f} ms/step (incl. teacher download/startup)')"
echo "===================================================="

# Post-run contract check: the distilled encoder.* must load into the CTC model.
CKPT="${OUTPUT_DIR}/checkpoint_step$(printf '%06d' ${MAX_STEPS})/checkpoint.pt"
echo ""
echo "Verifying checkpoint loads into CausalSpecUnitCTC: ${CKPT}"
python - "$CKPT" <<'PY'
import sys, os, torch
ckpt = sys.argv[1]
assert os.path.isfile(ckpt), f"Smoke did not produce a checkpoint at {ckpt}"
from CausalSpecUnit.model import CausalSpecUnitCTC
ctc = CausalSpecUnitCTC(vocab_size=128, variant="xs")
missing, unexpected = ctc.load_ssl_encoder(ckpt)
enc_keys = set(dict(ctc.encoder.named_parameters()).keys()) | set(dict(ctc.encoder.named_buffers()).keys())
enc_missing = [k for k in missing if k in enc_keys]
print(f"  load_ssl_encoder: missing={len(missing)} (encoder-missing={len(enc_missing)}) unexpected={len(unexpected)}")
assert len(unexpected) == 0, f"UNEXPECTED encoder keys — distill/CTC encoders diverged: {unexpected[:5]}"
assert len(enc_missing) == 0, f"CTC encoder keys NOT filled by distilled checkpoint: {enc_missing[:5]}"
print("  CONTRACT OK: distilled encoder loads cleanly into the CTC model.")
PY

echo ""
echo "===================================================="
echo "SMOKE PASSED — safe to submit the full run:"
echo "  TEACHER=${TEACHER} sbatch slurm/causal_specunit/50_distill_pretrain.sh"
echo "===================================================="
