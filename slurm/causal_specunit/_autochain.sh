# shellcheck shell=bash
# _autochain.sh — sourced by long pretraining scripts so a single run can
# survive an 4-hour cluster wall-time cap and complete across a chain of jobs.
#
# The cluster now caps every job at 4h (QOS bg-eng-01 MaxWall), but SSL /
# distillation pretraining needs far more GPU-time than that. This helper,
# sourced right before the torchrun launch, makes each 4h job:
#   1. resolve the latest checkpoint_step<N>/ already on disk in OUTPUT_DIR;
#   2. exit 0 immediately if training has already reached MAX_STEPS
#      (this is what terminates the chain);
#   3. auto-resume from that latest checkpoint (unless the caller pinned
#      RESUME_CKPT) by exporting RESUME_CKPT for the caller's own --resume logic;
#   4. queue ONE successor job with --dependency=afterany:<this job>, so training
#      picks up where it left off after this job ends for ANY reason (timeout,
#      preemption, or a clean finish — a clean finish just makes the successor
#      hit the step-2 terminator and exit).
#
# The caller MUST have already set (plain shell vars are fine):
#   OUTPUT_DIR   directory where checkpoint_step<N>/ dirs are written
#   MAX_STEPS    target optimizer steps; the chain stops once a checkpoint >= this
#   SELF_SCRIPT  repo-relative path of the caller, used to resubmit successors
#
# Optional env knobs:
#   AUTO_CHAIN=0    disable self-resubmission (run a single standalone 4h job)
#   CHAIN_MAX=150   hard cap on chain length; guards against a crash-loop
#   CHAIN_DEPTH    managed automatically (current link index; do not set by hand)

: "${OUTPUT_DIR:?_autochain.sh requires OUTPUT_DIR}"
: "${MAX_STEPS:?_autochain.sh requires MAX_STEPS}"
: "${SELF_SCRIPT:?_autochain.sh requires SELF_SCRIPT (repo-relative path of caller)}"

# --- 1. resolve the latest complete checkpoint --------------------------------
_ac_latest_step=-1
_ac_latest_ckpt=""
if [ -d "${OUTPUT_DIR}" ]; then
    for _ac_d in "${OUTPUT_DIR}"/checkpoint_step*; do
        [ -f "${_ac_d}/checkpoint.pt" ] || continue
        _ac_s="$(basename "${_ac_d}")"
        _ac_s="${_ac_s#checkpoint_step}"
        case "${_ac_s}" in ''|*[!0-9]*) continue ;; esac
        _ac_s=$((10#${_ac_s}))
        if [ "${_ac_s}" -gt "${_ac_latest_step}" ]; then
            _ac_latest_step=${_ac_s}
            _ac_latest_ckpt="${_ac_d}"
        fi
    done
fi

# --- 2. chain terminator: already trained to (or past) the target -------------
if [ "${_ac_latest_step}" -ge "${MAX_STEPS}" ]; then
    echo "[autochain] target reached: latest checkpoint step ${_ac_latest_step} >= MAX_STEPS ${MAX_STEPS}."
    echo "[autochain] nothing to do — exiting without queuing a successor. Chain complete."
    exit 0
fi

# --- 3. auto-resume from the latest checkpoint --------------------------------
if [ -z "${RESUME_CKPT:-}" ] && [ -n "${_ac_latest_ckpt}" ]; then
    export RESUME_CKPT="${_ac_latest_ckpt}"
    echo "[autochain] auto-resuming from ${RESUME_CKPT} (step ${_ac_latest_step} -> ${MAX_STEPS})"
elif [ -n "${RESUME_CKPT:-}" ]; then
    echo "[autochain] caller pinned RESUME_CKPT=${RESUME_CKPT}; honoring it"
else
    echo "[autochain] no checkpoint found in ${OUTPUT_DIR}; starting fresh (0 -> ${MAX_STEPS})"
fi

# --- 4. queue the successor (afterany) ----------------------------------------
CHAIN_DEPTH="${CHAIN_DEPTH:-0}"
# Guard against crash-loops. Sized for the longest legit run: m95 400k steps at
# a 4h QOS cap is ~35-45 links, so 150 leaves ample margin for timeouts/preemptions.
CHAIN_MAX="${CHAIN_MAX:-150}"
if [ "${AUTO_CHAIN:-1}" != "1" ]; then
    echo "[autochain] AUTO_CHAIN disabled; running a single standalone job (no successor queued)."
elif [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "[autochain] not under SLURM (no SLURM_JOB_ID); running once without chaining."
elif [ "${CHAIN_DEPTH}" -ge "${CHAIN_MAX}" ]; then
    echo "[autochain] CHAIN_DEPTH ${CHAIN_DEPTH} hit CHAIN_MAX ${CHAIN_MAX}; not queuing a successor."
    echo "[autochain] if training is not finished, raise CHAIN_MAX or investigate why it is not progressing."
else
    _ac_next=$((CHAIN_DEPTH + 1))
    # afterany so the successor runs whether this job times out, is preempted, or
    # finishes cleanly. RESUME_CKPT/MASTER_PORT are cleared so the successor
    # re-resolves the latest checkpoint and picks a fresh port from its own job id.
    _ac_nid="$(sbatch --parsable \
        --dependency="afterany:${SLURM_JOB_ID}" \
        --kill-on-invalid-dep=yes \
        --export="ALL,CHAIN_DEPTH=${_ac_next},RESUME_CKPT=,MASTER_PORT=" \
        "${SELF_SCRIPT}" 2>/dev/null || true)"
    if [ -n "${_ac_nid}" ]; then
        echo "[autochain] queued successor job ${_ac_nid} (link ${_ac_next}/${CHAIN_MAX}, afterany:${SLURM_JOB_ID})"
    else
        echo "[autochain] WARNING: failed to queue successor job — this job will run but the chain will not continue automatically."
    fi
fi
