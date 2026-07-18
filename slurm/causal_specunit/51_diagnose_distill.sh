#!/bin/bash
# Distillation collapse probe (diagnostic only — trains nothing, ~1-2 min/GPU).
#
# scripts/diagnose_distill.py needs a GPU (frozen 95M teacher forward + 9M
# student forward), so it can't run on the login node. This wraps it for one
# GPU on a compute node and scores the distilled student against trivial
# baselines on the SAME frames:
#   student    1 - cos(student_pred, teacher)        <- what training reported
#   mean-pred  1 - cos(global_mean_teacher, teacher) <- predict a constant
#   shuffled   1 - cos(teacher[perm], teacher)       <- a random OTHER frame
#   std ratio  student temporal std / teacher's      <- ~0 == constant output
#
# Read it:
#   student ~= mean-pred (std ratio ~0)  -> COLLAPSED. 0.145 was an illusion of
#     the anisotropic feature space; fix the objective (mean-center / layer-norm
#     the teacher targets before the loss) and re-run.
#   student << mean-pred                 -> distill is fine; look downstream.
#
# Submit (HuBERT run):
#   sbatch slurm/causal_specunit/51_diagnose_distill.sh
# wav2vec2 run (same checkpoint-dir convention):
#   TEACHER=wav2vec2_base sbatch slurm/causal_specunit/51_diagnose_distill.sh
# Both in one job:
#   TEACHERS="hubert_base wav2vec2_base" sbatch slurm/causal_specunit/51_diagnose_distill.sh
#
# Optional knobs:
#   STEP=250000 CKPT=<explicit/path/checkpoint.pt> SPLIT=dev-clean NUM_UTTS=64 sbatch ...

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=distill_probe
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/distill_probe.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/distill_probe.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
# CMVN must match what the student was distilled/fine-tuned with (targets_960h_c8).
SOURCE_TARGETS_DIR="${SOURCE_TARGETS_DIR:-outputs/causal_specunit/targets_960h_c8}"

# One or more teachers to probe. TEACHER (single) is honored for symmetry with
# the 50_* scripts; TEACHERS (list) overrides it.
TEACHERS="${TEACHERS:-${TEACHER:-hubert_base}}"
TEACHER_LAYERS="${TEACHER_LAYERS:-3 7 11}"
STEP="${STEP:-250000}"
SPLIT="${SPLIT:-dev-clean}"
NUM_UTTS="${NUM_UTTS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VARIANT="${VARIANT:-xs}"

# Teacher checkpoint cache (same location the 50_* runs used).
export TORCH_HOME="${TORCH_HOME:-${PROJECT_DIR}/.torch_cache}"

# Sanity checks
[ -d "${VIRTUAL_ENV}" ]                 || { echo "Missing venv: ${VIRTUAL_ENV}"; exit 1; }
[ -d "${DATA_ROOT}" ]                   || { echo "Missing data root: ${DATA_ROOT}"; exit 1; }
[ -f "${SOURCE_TARGETS_DIR}/cmvn.pt" ]  || { echo "Missing CMVN: ${SOURCE_TARGETS_DIR}/cmvn.pt"; exit 1; }

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
mkdir -p logs "${TORCH_HOME}"

STEP_PADDED="$(printf '%06d' "${STEP}")"

for TEACHER in ${TEACHERS}; do
    case "${TEACHER}" in
        hubert_base|wav2vec2_base) ;;
        *) echo "TEACHER must be hubert_base or wav2vec2_base, got: ${TEACHER}"; exit 1 ;;
    esac

    # Same checkpoint-dir convention as 50_distill_pretrain.sh:
    #   outputs/causal_specunit/distill_<teacher>_960h/checkpoint_step<NNNNNN>/checkpoint.pt
    CKPT_DEFAULT="outputs/causal_specunit/distill_${TEACHER}_960h/checkpoint_step${STEP_PADDED}/checkpoint.pt"
    CKPT="${CKPT:-${CKPT_DEFAULT}}"

    echo ""
    echo "===================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DISTILL PROBE: ${TEACHER}"
    echo "  checkpoint = ${CKPT}"
    echo "  cmvn=${SOURCE_TARGETS_DIR}/cmvn.pt  split=${SPLIT}  utts=${NUM_UTTS}"
    echo "===================================================="

    if [ ! -f "${CKPT}" ]; then
        echo "  !! checkpoint not found: ${CKPT}"
        echo "  !! set CKPT=<path> or STEP=<n>, or check the run finished. Skipping ${TEACHER}."
        unset CKPT
        continue
    fi

    python scripts/diagnose_distill.py \
        --checkpoint "${CKPT}" \
        --data-root "${DATA_ROOT}" \
        --cmvn-path "${SOURCE_TARGETS_DIR}/cmvn.pt" \
        --split "${SPLIT}" \
        --teacher "${TEACHER}" \
        --teacher-layers ${TEACHER_LAYERS} \
        --variant "${VARIANT}" \
        --num-utts "${NUM_UTTS}" \
        --batch-size "${BATCH_SIZE}"

    # Reset so a per-teacher default is recomputed on the next loop iteration.
    unset CKPT
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] probe done."
