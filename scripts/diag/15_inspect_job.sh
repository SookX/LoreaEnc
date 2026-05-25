#!/bin/bash
# Inspect a running (or hung) slurm job: status, node, GPU usage,
# python process list, and Python stack traces via py-spy when available.
#
# Run on the login node:
#   bash scripts/diag/15_inspect_job.sh <JOBID>
#
# Reads JOBID from $1 or from $JOBID env. Tries py-spy at three locations:
# the venv that the slurm jobs use, the user pip --user dir, and PATH.
# If py-spy is missing, falls back to /proc/<pid>/stack via SSH.

set -uo pipefail

JOBID="${1:-${JOBID:-}}"
if [ -z "${JOBID}" ]; then
    echo "Usage: $0 <JOBID>   (or set JOBID=<n> in the environment)"
    exit 1
fi

VENV="${VENV:-/valhalla/projects/bg-eng-01/conda_envs/torch}"
PYSPY_CANDIDATES=(
    "${VENV}/bin/py-spy"
    "${HOME}/.local/bin/py-spy"
    "py-spy"
)
PYSPY=""
for c in "${PYSPY_CANDIDATES[@]}"; do
    if command -v "${c}" >/dev/null 2>&1; then
        PYSPY="${c}"
        break
    fi
done

echo "========================================"
echo "Slurm job ${JOBID} status"
echo "========================================"
squeue -j "${JOBID}" 2>/dev/null || true
echo
scontrol show job "${JOBID}" | head -40 || true

NODE=$(scontrol show job "${JOBID}" -o 2>/dev/null | tr ' ' '\n' | awk -F= '/^NodeList=/{print $2}' | head -1)
if [ -z "${NODE}" ] || [ "${NODE}" = "(null)" ]; then
    echo "Job ${JOBID} has no allocated node (yet?). Exiting."
    exit 0
fi
echo
echo "Allocated node: ${NODE}"

INSPECT_REMOTE=$(cat <<'REMOTE'
echo
echo "---- GPU utilisation (nvidia-smi) ----"
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"
echo
echo "---- Python processes (top) ----"
ps -o pid,etime,stat,pcpu,pmem,wchan:24,cmd -u "$USER" 2>/dev/null | grep -E "python|torchrun|train_ctc" | grep -v grep | head -20
echo
echo "---- /proc/<pid>/wchan (where each python is sleeping) ----"
for pid in $(pgrep -u "$USER" -f "train_ctc|pretrain_ssl" 2>/dev/null | head -10); do
    wchan=$(cat /proc/${pid}/wchan 2>/dev/null || echo "?")
    state=$(awk '/^State:/ {print $2 $3}' /proc/${pid}/status 2>/dev/null || echo "?")
    echo "  pid=${pid} state=${state} wchan=${wchan}"
done
REMOTE
)

ssh -o ConnectTimeout=10 -o BatchMode=yes "${NODE}" "${INSPECT_REMOTE}" || {
    echo "SSH to ${NODE} failed; cannot inspect remotely."
    exit 0
}

if [ -n "${PYSPY}" ]; then
    echo
    echo "---- py-spy stack dumps (top of stack for each python on ${NODE}) ----"
    ssh -o ConnectTimeout=10 -o BatchMode=yes "${NODE}" "
        for pid in \$(pgrep -u \$USER -f 'train_ctc|pretrain_ssl' 2>/dev/null | head -10); do
            echo
            echo '--- py-spy dump pid=' \$pid '---'
            ${PYSPY} dump --pid \$pid 2>&1 | head -30
        done
    " || true
else
    echo
    echo "py-spy not found in any candidate path. Install with:"
    echo "  ${VENV}/bin/pip install py-spy"
    echo "Or with --user:"
    echo "  pip install --user py-spy"
    echo "Then rerun this script."
fi
