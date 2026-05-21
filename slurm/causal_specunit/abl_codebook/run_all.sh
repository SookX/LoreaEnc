#!/bin/bash
# Launcher for the dual-codebook ablation.
#
# Submits 15 jobs total, chained with --dependency=afterok:
#   For each codebook in {coarse, fine, both}:
#     SSL_<mode>  -> FT_<mode>_10h  -> EVAL_<mode>_10h
#                 \> FT_<mode>_100h -> EVAL_<mode>_100h
#
# SLURM will execute as many jobs in parallel as your allocation allows. If
# you have a single 4-GPU slot, jobs serialize and the whole thing takes
# ~75h. With three concurrent 4-GPU slots, ~20h.
#
# After everything completes, run scripts/abl_codebook_table.py to print a
# LaTeX table ready to paste.

set -euo pipefail

if [ -z "${SLURM_JOB_ACCOUNT:-}" ]; then
    SLURM_JOB_ACCOUNT="bg-eng-01"
fi

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
SLURM_DIR="${PROJECT_DIR}/slurm/causal_specunit/abl_codebook"
OUT_BASE="outputs/causal_specunit/abl_codebook"

cd "${PROJECT_DIR}"

declare -A SSL_JOB
declare -A FT_JOB
declare -A EVAL_JOB

submit() {
    # Print the command on stderr for traceability, then sbatch.
    echo "[submit] $*" >&2
    "$@"
}

for mode in coarse fine both; do
    SSL_OUT="${OUT_BASE}/ssl_${mode}"
    SSL_CKPT="${SSL_OUT}/checkpoint_step050000/checkpoint.pt"

    # ---- SSL pretrain ----
    SSL_JOB[$mode]=$(submit sbatch --parsable \
        --export=ALL,CODEBOOK_MODE=${mode},SSL_OUTPUT_DIR=${SSL_OUT} \
        "${SLURM_DIR}/01_ssl.sh")
    echo "SSL ${mode}: ${SSL_JOB[$mode]}"

    # ---- Fine-tunes (depend on SSL) ----
    for hours in 10 100; do
        FT_OUT="${OUT_BASE}/ft_${mode}_${hours}h"
        FT_KEY="${mode}_${hours}"
        FT_JOB[$FT_KEY]=$(submit sbatch --parsable \
            --dependency=afterok:${SSL_JOB[$mode]} \
            --export=ALL,SSL_CHECKPOINT=${SSL_CKPT},TRAIN_HOURS=${hours},FT_OUTPUT_DIR=${FT_OUT} \
            "${SLURM_DIR}/02_ft.sh")
        echo "  FT ${mode} ${hours}h: ${FT_JOB[$FT_KEY]} (after SSL ${SSL_JOB[$mode]})"

        # ---- Test eval (depends on its fine-tune) ----
        EVAL_JOB[$FT_KEY]=$(submit sbatch --parsable \
            --dependency=afterok:${FT_JOB[$FT_KEY]} \
            --export=ALL,FT_OUTPUT_DIR=${FT_OUT} \
            "${SLURM_DIR}/03_eval.sh")
        echo "  Eval ${mode} ${hours}h: ${EVAL_JOB[$FT_KEY]} (after FT ${FT_JOB[$FT_KEY]})"
    done
done

# Print a summary of every job ID we submitted, for monitoring.
echo ""
echo "All 15 jobs submitted. Monitor with:"
JOB_IDS=""
for k in "${!SSL_JOB[@]}";  do JOB_IDS="${JOB_IDS}${SSL_JOB[$k]},"; done
for k in "${!FT_JOB[@]}";   do JOB_IDS="${JOB_IDS}${FT_JOB[$k]},"; done
for k in "${!EVAL_JOB[@]}"; do JOB_IDS="${JOB_IDS}${EVAL_JOB[$k]},"; done
JOB_IDS="${JOB_IDS%,}"
echo "  squeue -u \$USER -j ${JOB_IDS}"
echo ""
echo "When the 6 eval jobs are all DONE, run:"
echo "  python scripts/abl_codebook_table.py --base-dir ${OUT_BASE}"
