#!/bin/bash
# Download + extract Multilingual LibriSpeech (MLS) languages for the
# cross-language transfer experiment (SALMA resubmission, critique #2).
#
# Defaults to the two small, low-resource languages we chose over French/German:
#   Polish (~103h train, 6.6 GB flac) and Portuguese (~161h, 10 GB flac).
# Both are the same audiobook domain and standard-split structure as English
# LibriSpeech, so the existing target-gen / pretrain / fine-tune scripts reuse
# cleanly once these are on disk.
#
# Submit (both defaults):
#   sbatch slurm/causal_specunit/60_download_mls.sh
# One language, or a different set:
#   LANGS="polish" sbatch slurm/causal_specunit/60_download_mls.sh
#   LANGS="italian spanish" sbatch slurm/causal_specunit/60_download_mls.sh
# Opus (lossy, ~4x smaller) instead of flac — NOT recommended for our pipeline:
#   VARIANT=_opus sbatch slurm/causal_specunit/60_download_mls.sh
#
# Resumable: wget -c continues a partial tarball, so just re-submit if a job
# times out. Extraction is skipped if the language dir already looks complete.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=mls_download
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# The bg-eng-01 QOS mandates a GPU per job (QOSMinGRES), so request the minimum
# even though downloading/extracting uses none.
#SBATCH --gres=gpu:1
#SBATCH -o /valhalla/projects/bg-eng-01/LoreaEnc/logs/mls_download.%j.out
#SBATCH -e /valhalla/projects/bg-eng-01/LoreaEnc/logs/mls_download.%j.err

set -euo pipefail

PROJECT_DIR="/valhalla/projects/${SLURM_JOB_ACCOUNT:-bg-eng-01}/LoreaEnc"
cd "${PROJECT_DIR}"
mkdir -p logs

DEST="${DEST:-dataset/mls}"
LANGS="${LANGS:-polish portuguese}"
VARIANT="${VARIANT:-}"           # "" = flac (default), "_opus" = lossy/smaller
BASE_URL="https://dl.fbaipublicfiles.com/mls"
KEEP_TAR="${KEEP_TAR:-1}"        # keep the tarball after extract (1) or delete (0)

mkdir -p "${DEST}"

echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] MLS download"
echo "  dest=${DEST}  langs=[${LANGS}]  variant='${VARIANT:-flac}'"
echo "===================================================="

for LANG in ${LANGS}; do
    NAME="mls_${LANG}${VARIANT}"
    URL="${BASE_URL}/${NAME}.tar.gz"
    TAR="${DEST}/${NAME}.tar.gz"
    # The tarball's top-level dir is always mls_<lang> (no _opus suffix inside).
    EXTRACT_DIR="${DEST}/mls_${LANG}"

    echo ""
    echo "---- ${LANG} ----"
    echo "  url=${URL}"

    if [ -d "${EXTRACT_DIR}/train/audio" ] && [ -f "${EXTRACT_DIR}/train/transcripts.txt" ]; then
        echo "  already extracted at ${EXTRACT_DIR} (train/audio + transcripts present) — skipping."
        continue
    fi

    # Fail fast if the URL is gone, before a long wget.
    CODE=$(curl -so /dev/null -w "%{http_code}" -IL --max-time 60 "${URL}" || echo "000")
    if [ "${CODE}" != "200" ]; then
        echo "  !! ${URL} returned HTTP ${CODE} — skipping ${LANG}."
        continue
    fi

    echo "  [$(date '+%H:%M:%S')] downloading (resumable)..."
    wget -c --tries=10 --timeout=120 --waitretry=10 -O "${TAR}" "${URL}"

    echo "  [$(date '+%H:%M:%S')] verifying archive integrity..."
    if ! tar tzf "${TAR}" >/dev/null 2>&1; then
        echo "  !! ${TAR} is corrupt/incomplete. Re-submit to resume the download."
        exit 1
    fi

    echo "  [$(date '+%H:%M:%S')] extracting into ${DEST}/ ..."
    tar xzf "${TAR}" -C "${DEST}"

    if [ ! -d "${EXTRACT_DIR}/train/audio" ]; then
        echo "  !! expected ${EXTRACT_DIR}/train/audio after extract — layout differs, inspect manually."
        exit 1
    fi

    if [ "${KEEP_TAR}" = "0" ]; then
        echo "  removing tarball ${TAR} (KEEP_TAR=0)"
        rm -f "${TAR}"
    fi

    echo "  [$(date '+%H:%M:%S')] done: ${EXTRACT_DIR}"
    du -sh "${EXTRACT_DIR}" 2>/dev/null || true
done

echo ""
echo "===================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] MLS download job finished."
echo "Layout under ${DEST}:"
for LANG in ${LANGS}; do
    d="${DEST}/mls_${LANG}"
    [ -d "${d}" ] && { echo "  ${d}:"; ls "${d}" 2>/dev/null | sed 's/^/    /'; }
done
echo "Next: preprocessing (target-gen / manifests) — decide once the audio is on disk."
echo "===================================================="
