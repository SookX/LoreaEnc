#!/bin/bash
# Dual-codebook ablation — ONE SLURM JOB that runs the entire pipeline.
#
# Sequential phases inside this single allocation:
#   1.  SSL pretrain × 3      ({coarse, fine, both} × 50k steps)        ~6h each = 18h
#   2.  CTC fine-tune × 6     (3 codebooks × {10h, 100h} × 100 epochs)
#        - 100h legs ~12h each = 36h
#        - 10h legs  ~3h each  = 9h
#   3.  Test eval × 6         (test-clean + test-other on best ckpt)    ~30min each = 3h
#   4.  Aggregate to LaTeX                                              ~1 min
#
#   Sequential total wall clock ≈ 65–75h on 4 GPUs.
#
# IDEMPOTENT: every phase checks whether its output already exists and skips
# if so. If the job dies (or hits the time limit), just resubmit — it will
# resume from the first unfinished phase.
#
# Override knobs (env vars at sbatch time):
#   MAX_STEPS              SSL steps per codebook         (default 50000)
#   EPOCHS                 CTC fine-tune epochs           (default 100)
#   SKIP_10H=1             omit the 10h fine-tune leg (saves ~9h)
#   SKIP_100H=1            omit the 100h fine-tune leg (saves ~36h)
#   ONLY_MODES="coarse"    run only specific codebooks (space-separated)

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=csu_abl_codebook
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_abl_codebook.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/csu_abl_codebook.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc"
VIRTUAL_ENV="/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch"
DATA_ROOT="dataset/datasets/librispeech/LibriSpeech"
TARGETS_DIR="outputs/causal_specunit/targets_960h_c8"
TOKENIZER_PATH="dataset/bpe128.model"
OUT_BASE="outputs/causal_specunit/abl_codebook"

MAX_STEPS="${MAX_STEPS:-50000}"
EPOCHS="${EPOCHS:-100}"
SKIP_10H="${SKIP_10H:-0}"
SKIP_100H="${SKIP_100H:-0}"
ONLY_MODES="${ONLY_MODES:-coarse fine both}"

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "${PROJECT_DIR}"
mkdir -p logs "${OUT_BASE}"

# Sanity checks
[ -d "${DATA_ROOT}" ]                  || { echo "Missing data: ${DATA_ROOT}"; exit 1; }
[ -f "${TARGETS_DIR}/cmvn.pt" ]        || { echo "Missing CMVN"; exit 1; }
[ -f "${TARGETS_DIR}/targets.pt" ]     || { echo "Missing targets"; exit 1; }
[ -f "${TARGETS_DIR}/metadata.json" ]  || { echo "Missing target metadata"; exit 1; }
[ -f "${TOKENIZER_PATH}" ]             || { echo "Missing tokenizer"; exit 1; }
if [ ! -f "${TARGETS_DIR}/target_index.json" ]; then
    python -m CausalSpecUnit.shard_targets --targets-dir "${TARGETS_DIR}" --num-shards 128
fi

# Distinct master ports so phases don't collide if a previous run left state around.
PORT_BASE=$((16000 + SLURM_JOB_ID % 10000))
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

log_phase() {
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "============================================================"
}

# ---------- Phase 1: SSL pretrain × 3 ----------
run_ssl() {
    local mode="$1"
    local ssl_out="${OUT_BASE}/ssl_${mode}"
    local ssl_ckpt="${ssl_out}/checkpoint_step$(printf '%06d' ${MAX_STEPS})/checkpoint.pt"
    mkdir -p "${ssl_out}"

    if [ -f "${ssl_ckpt}" ]; then
        log_phase "SKIP SSL ${mode} — checkpoint already exists: ${ssl_ckpt}"
        return 0
    fi

    log_phase "SSL ${mode}  |  ${MAX_STEPS} steps  |  4 GPUs"
    export MASTER_PORT=$((PORT_BASE + 100))
    torchrun \
        --nproc_per_node=4 \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m CausalSpecUnit.pretrain_ssl \
        --data-root "${DATA_ROOT}" \
        --targets-dir "${TARGETS_DIR}" \
        --output-dir "${ssl_out}" \
        --variant xs \
        --epochs 1000 \
        --max-steps "${MAX_STEPS}" \
        --batch-size 64 \
        --grad-accum-steps 1 \
        --codebook-mode "${mode}" \
        --mask-prob 0.50 \
        --mask-length 10 \
        --chunk-size 8 \
        --chunk-stride 4 \
        --layer-drop-p 0.05 \
        --aux-layers 8 \
        --aux-weight 0.5 \
        --specaug \
        --specaug-time-mask-param 30 \
        --specaug-freq-mask-param 20 \
        --specaug-time-masks 2 \
        --specaug-freq-masks 2 \
        --lr 1e-3 \
        --warmup-epochs 10 \
        --peak-epochs 10 \
        --noam-decay-rate 1.0 \
        --max-grad-norm 1.0 \
        --max-safe-grad-norm 200.0 \
        --workers 8 \
        --dataloader-timeout 300 \
        --prefetch-factor 4 \
        --log-every 100 \
        --save-every 5 \
        --progress off
}

# ---------- Phase 2: CTC fine-tune × {hours} ----------
run_ft() {
    local mode="$1"
    local hours="$2"
    local ssl_ckpt="${OUT_BASE}/ssl_${mode}/checkpoint_step$(printf '%06d' ${MAX_STEPS})/checkpoint.pt"
    local ft_out="${OUT_BASE}/ft_${mode}_${hours}h"
    local ft_best="${ft_out}/checkpoint_best/checkpoint.pt"
    mkdir -p "${ft_out}"

    if [ -f "${ft_best}" ]; then
        log_phase "SKIP FT ${mode}_${hours}h — checkpoint_best exists: ${ft_best}"
        return 0
    fi
    [ -f "${ssl_ckpt}" ] || { echo "[error] missing SSL checkpoint ${ssl_ckpt}"; return 1; }

    local subset=""
    if [ "${hours}" = "10" ]; then
        subset="--train-subset-hours 10 --train-subset-seed 42"
    fi

    log_phase "FT ${mode}_${hours}h  |  ${EPOCHS} epochs  |  SSL=${ssl_ckpt}  |  recipe=anchored"
    export MASTER_PORT=$((PORT_BASE + 200 + (hours == 10 ? 1 : 2)))
    # Anchored fine-tune recipe (the strongest one we have):
    # LP-FT rewarmup + layer-decay 0.85 + InterCTC@7 (w=0.15) + ssl-mask SpecAug
    # + SSL anchor (w=0.1, heads warm-started from the SSL k-means heads).
    # IDENTICAL across all three codebook conditions — only the SSL pretraining
    # objective varies. Anchor uses targets from ${TARGETS_DIR} (the original
    # iter-1 k-means targets); the encoder state — including which of K=100 /
    # K=500 / both it was trained on — is the only thing that differs.
    torchrun \
        --nproc_per_node=4 \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m CausalSpecUnit.train_ctc \
        --data-root "${DATA_ROOT}" \
        --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --train-splits train-clean-100 \
        ${subset} \
        --ssl-checkpoint "${ssl_ckpt}" \
        --output-dir "${ft_out}" \
        --variant xs \
        --epochs "${EPOCHS}" \
        --batch-size 128 \
        --grad-accum-steps 1 \
        --eval-batch-size 128 \
        --eval-split dev-other \
        --eval-every 1 \
        --workers 8 \
        --dataloader-timeout 120 \
        --lr 1e-3 \
        --encoder-lr 6e-4 \
        --head-lr 1e-3 \
        --encoder-layer-lr-decay 0.85 \
        --no-decay-norm-and-bias \
        --freeze-encoder-epochs 0 \
        --encoder-rewarmup-epochs 5 \
        --inter-ctc-layers 7 \
        --inter-ctc-weight 0.15 \
        --specaug-mask-source ssl-mask \
        --ssl-anchor-weight 0.1 \
        --ssl-anchor-targets-dir "${TARGETS_DIR}" \
        --ssl-anchor-load-heads \
        --warmup-epochs 10 \
        --peak-epochs 50 \
        --noam-decay-rate 0.5 \
        --max-grad-norm 1.0 \
        --specaug \
        --specaug-time-mask-param 30 \
        --specaug-freq-mask-param 20 \
        --specaug-time-masks 2 \
        --specaug-freq-masks 2 \
        --specaug-disable-last-epochs 10 \
        --progress off \
        --log-every 0 \
        --save-every 10
}

# ---------- Phase 3: test eval × 6 ----------
run_eval() {
    local mode="$1"
    local hours="$2"
    local ft_out="${OUT_BASE}/ft_${mode}_${hours}h"
    local ckpt="${ft_out}/checkpoint_best/checkpoint.pt"
    local results="${ft_out}/eval_results.json"

    if [ -f "${results}" ]; then
        log_phase "SKIP EVAL ${mode}_${hours}h — results exist: ${results}"
        return 0
    fi
    [ -f "${ckpt}" ] || { echo "[error] missing FT checkpoint ${ckpt}"; return 1; }

    log_phase "EVAL ${mode}_${hours}h  |  test-clean + test-other"
    export MASTER_PORT=$((PORT_BASE + 300))
    torchrun \
        --nproc_per_node=4 \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m CausalSpecUnit.evaluate_ctc \
        --checkpoint "${ckpt}" \
        --data-root "${DATA_ROOT}" \
        --cmvn-path "${TARGETS_DIR}/cmvn.pt" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --variant xs \
        --splits test-clean test-other \
        --batch-size 64 \
        --workers 4 \
        --output "${results}"
}

# ---------- Driver ----------
log_phase "BEGIN dual-codebook ablation (job ${SLURM_JOB_ID})"
echo "modes:      ${ONLY_MODES}"
echo "ssl steps:  ${MAX_STEPS}"
echo "ft epochs:  ${EPOCHS}"
echo "skip 10h:   ${SKIP_10H}"
echo "skip 100h:  ${SKIP_100H}"
echo "out base:   ${OUT_BASE}"

# Phase 1: all SSLs first (they're independent of each other).
for mode in ${ONLY_MODES}; do
    run_ssl "${mode}"
done

# Phase 2 + 3 interleaved per codebook: FT then immediate eval, so the eval
# runs while disk caches are warm and we get partial table data as soon as
# possible.
for mode in ${ONLY_MODES}; do
    if [ "${SKIP_10H}" != "1" ]; then
        run_ft "${mode}" 10
        run_eval "${mode}" 10
    fi
    if [ "${SKIP_100H}" != "1" ]; then
        run_ft "${mode}" 100
        run_eval "${mode}" 100
    fi
done

# Phase 4: aggregate to LaTeX. Always emits (with placeholders for any cells
# that didn't complete), so partial runs still produce inspectable output.
log_phase "AGGREGATE  →  LaTeX table"
python scripts/abl_codebook_table.py \
    --base-dir "${OUT_BASE}" \
    --latex-out "${OUT_BASE}/abl_codebook.tex" || true

log_phase "DONE  (job ${SLURM_JOB_ID})"
echo "Table:   ${OUT_BASE}/abl_codebook.tex"
echo "All eval_results.json files:"
find "${OUT_BASE}" -name eval_results.json -print 2>/dev/null || true
