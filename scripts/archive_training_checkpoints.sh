#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 LOCAL_RUNS_ROOT ARCHIVE_ROOT [--once]" >&2
}

LOCAL_RUNS_ROOT="${1:-}"
ARCHIVE_ROOT="${2:-}"
MODE="${3:---watch}"
INTERVAL_SEC="${CHECKPOINT_ARCHIVE_INTERVAL_SEC:-30}"

if [[ -z "${LOCAL_RUNS_ROOT}" || -z "${ARCHIVE_ROOT}" ]]; then
  usage
  exit 2
fi
if [[ "${MODE}" != "--watch" && "${MODE}" != "--once" ]]; then
  usage
  exit 2
fi
if ! [[ "${INTERVAL_SEC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CHECKPOINT_ARCHIVE_INTERVAL_SEC must be a positive integer" >&2
  exit 2
fi

mkdir -p "${LOCAL_RUNS_ROOT}" "${ARCHIVE_ROOT}"
LOCAL_RUNS_ROOT="$(realpath "${LOCAL_RUNS_ROOT}")"
ARCHIVE_ROOT="$(realpath "${ARCHIVE_ROOT}")"
LOCK_FILE="${LOCAL_RUNS_ROOT}/.checkpoint-archiver.lock"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "ERROR: another checkpoint archiver is already using ${LOCAL_RUNS_ROOT}" >&2
  exit 1
fi

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"
}

checkpoint_directories() {
  find "${LOCAL_RUNS_ROOT}" -mindepth 4 -maxdepth 4 -type d -path '*/output/checkpoints/*' -print0
}

archive_completed_checkpoints() {
  local checkpoint step checkpoints_dir output_dir job archive_job partial destination verification
  while IFS= read -r -d '' checkpoint; do
    step="$(basename "${checkpoint}")"
    [[ "${step}" =~ ^[0-9]{6}$ ]] || continue
    test -f "${checkpoint}/training_state/training_step.json" || continue

    checkpoints_dir="$(dirname "${checkpoint}")"
    output_dir="$(dirname "${checkpoints_dir}")"
    job="$(basename "$(dirname "${output_dir}")")"
    archive_job="${ARCHIVE_ROOT}/${job}"
    destination="${archive_job}/${step}"
    partial="${archive_job}/.${step}.partial"

    if test -f "${destination}/ARCHIVE_COMPLETE"; then
      continue
    fi
    if test -e "${destination}"; then
      log "ERROR destination exists without completion marker: ${destination}"
      continue
    fi

    mkdir -p "${archive_job}" "${partial}"
    log "COPY job=${job} checkpoint=${step} destination=${destination}"
    rsync -a --delete --partial "${checkpoint}/" "${partial}/"
    verification="$(
      rsync -a --delete --checksum --dry-run --itemize-changes \
        "${checkpoint}/" "${partial}/"
    )"
    if [[ -n "${verification}" ]]; then
      log "ERROR checksum verification failed for ${job}/${step}: ${verification}"
      continue
    fi

    cat > "${partial}/ARCHIVE_COMPLETE" <<EOF
job=${job}
checkpoint=${step}
archived_utc=$(date -u +%FT%TZ)
source=${checkpoint}
EOF
    mv "${partial}" "${destination}"
    log "ARCHIVED job=${job} checkpoint=${step}"
  done < <(checkpoint_directories)
}

remove_archived_nonlatest_checkpoints() {
  local checkpoints_dir job latest checkpoint step destination
  while IFS= read -r -d '' checkpoints_dir; do
    job="$(basename "$(dirname "$(dirname "${checkpoints_dir}")")")"
    latest="$(
      find "${checkpoints_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
        | grep -E '^[0-9]{6}$' | sort -n | tail -1
    )"
    [[ -n "${latest}" ]] || continue

    while IFS= read -r -d '' checkpoint; do
      step="$(basename "${checkpoint}")"
      [[ "${step}" =~ ^[0-9]{6}$ ]] || continue
      [[ "${step}" != "${latest}" ]] || continue
      destination="${ARCHIVE_ROOT}/${job}/${step}"
      test -f "${destination}/ARCHIVE_COMPLETE" || continue
      [[ "${checkpoint}" == "${LOCAL_RUNS_ROOT}/"*"/output/checkpoints/${step}" ]] || {
        log "ERROR refusing unexpected cleanup target: ${checkpoint}"
        continue
      }
      rm -rf -- "${checkpoint}"
      log "REMOVED_LOCAL job=${job} checkpoint=${step} retained_latest=${latest}"
    done < <(find "${checkpoints_dir}" -mindepth 1 -maxdepth 1 -type d -print0)
  done < <(
    find "${LOCAL_RUNS_ROOT}" -mindepth 3 -maxdepth 3 -type d -path '*/output/checkpoints' -print0
  )
}

run_once() {
  archive_completed_checkpoints
  remove_archived_nonlatest_checkpoints
}

if [[ "${MODE}" == "--once" ]]; then
  run_once
  exit 0
fi

log "START local=${LOCAL_RUNS_ROOT} archive=${ARCHIVE_ROOT} interval=${INTERVAL_SEC}s"
while true; do
  run_once
  sleep "${INTERVAL_SEC}"
done
