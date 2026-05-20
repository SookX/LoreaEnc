#!/bin/bash
# Submit v2 SSL pretraining, then submit CTC fine-tuning after pretraining succeeds.
#
# Run from the repo root on the cluster:
#   bash slurm/causal_specunit/05_submit_ssl_v2_150k_then_ctc.sh

set -euo pipefail

PRETRAIN_SCRIPT="${PRETRAIN_SCRIPT:-slurm/causal_specunit/02_pretrain_ssl_v2_100k_c8.sh}"
CTC_SCRIPT="${CTC_SCRIPT:-slurm/causal_specunit/03_train_ctc_150ep_fair_ssl.sh}"

MAX_STEPS="${MAX_STEPS:-150000}"
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-outputs/causal_specunit/pretrain_ssl_v2_150k_c8}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-${PRETRAIN_OUTPUT_DIR}/checkpoint_step150000/checkpoint.pt}"
CTC_OUTPUT_DIR="${CTC_OUTPUT_DIR:-outputs/causal_specunit/ctc_ssl_960h_v2_150k_elr4e4_ld100_hlr1e3_w10_p90_d025_150ep_c8}"

pretrain_jid="$(
    sbatch \
        --parsable \
        --export=ALL,MAX_STEPS="${MAX_STEPS}",OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR}" \
        "${PRETRAIN_SCRIPT}"
)"

ctc_jid="$(
    sbatch \
        --parsable \
        --dependency="afterok:${pretrain_jid}" \
        --export=ALL,SSL_CHECKPOINT="${PRETRAIN_CHECKPOINT}",OUTPUT_DIR="${CTC_OUTPUT_DIR}" \
        "${CTC_SCRIPT}"
)"

echo "Submitted SSL v2 pretraining job: ${pretrain_jid}"
echo "  output: ${PRETRAIN_OUTPUT_DIR}"
echo "  checkpoint: ${PRETRAIN_CHECKPOINT}"
echo "Submitted dependent CTC fine-tune job: ${ctc_jid}"
echo "  dependency: afterok:${pretrain_jid}"
echo "  output: ${CTC_OUTPUT_DIR}"
