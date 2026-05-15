#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_purity
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_purity.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_purity.%j.err

set -euo pipefail

module purge
module load anaconda3

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
TEXTGRID_DIR="${TEXTGRID_DIR:-CausalSpecUnit/librispeech_alignments}"
OUTPUT_PATH="outputs/causal_specunit/phone_purity_c8.npz"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1

cd "${PROJECT_DIR}"
mkdir -p logs outputs/causal_specunit

if [ ! -f "${TARGETS_DIR}/targets.pt" ]; then
    echo "Missing targets: ${TARGETS_DIR}/targets.pt"
    echo "Run slurm/causal_specunit/01_generate_targets.sh first."
    exit 1
fi

if [ ! -d "${TEXTGRID_DIR}" ]; then
    echo "Missing TextGrid directory: ${TEXTGRID_DIR}"
    echo "Submit with: TEXTGRID_DIR=/actual/path sbatch slurm/causal_specunit/01b_phone_purity.sh"
    exit 1
fi

python -m CausalSpecUnit.evaluate_phone_purity \
    --targets-dir "${TARGETS_DIR}" \
    --textgrid-dir "${TEXTGRID_DIR}" \
    --tier phones \
    --output "${OUTPUT_PATH}"

python -m CausalSpecUnit.evaluate_phone_purity \
    --targets-dir "${TARGETS_DIR}" \
    --textgrid-dir "${TEXTGRID_DIR}" \
    --tier phones \
    --exclude-silence \
    --output "outputs/causal_specunit/phone_purity_c8_no_silence.npz"
