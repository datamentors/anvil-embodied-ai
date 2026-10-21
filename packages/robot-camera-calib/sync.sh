#!/usr/bin/env bash
# Copy this package to a robot, fetch runs back, or tunnel the live view.
# On the robot the code lands where the repo would be: ~/<repo>/<this folder's path in
# the repo>, e.g. ~/anvil-embodied-ai/packages/robot-camera-calib. <repo> is the name
# of this checkout's git origin (REMOTE_REPO overrides it); REMOTE_CODE=<path under
# the robot user's home> overrides the whole path.
#
#   ./sync.sh push          code here -> robot (overwrites robot code; runs/ untouched)
#   ./sync.sh status        what push would change, changes nothing
#   ./sync.sh pull <run>    robot <runs_dir>/<run> -> ./runs/<run>  (runs_dir from robots/<TARGET>.yaml)
#   ./sync.sh view          tunnel robot :9701 -> localhost:9701 and open the live view
#
# TARGET=<robot> picks robots/<robot>.yaml (default g2). Login: ROBOT_HOST=user@ip
# [ROBOT_PASS=...], or for g2 / anvil the "# G2_Agi" / "# Anvil_calib" entry of the
# nearest instances.txt above this folder (or INSTANCES=<file>).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${TARGET:-g2}"
CONFIG="$HERE/robots/$TARGET.yaml"
[[ -f "$CONFIG" ]] || { echo "no $CONFIG" >&2; exit 1; }
if [[ -z "${REMOTE_CODE:-}" ]]; then
  # From git, so the same script works in any repo or worktree it is copied into.
  repo_name="$(basename "$(git -C "$HERE" remote get-url origin 2>/dev/null)" .git)"
  repo_path="$(git -C "$HERE" rev-parse --show-prefix 2>/dev/null)"
  [[ -n "${REMOTE_REPO:-$repo_name}" && -n "$repo_path" ]] || {
    echo "not in a git checkout with an origin: set REMOTE_CODE=<path under the robot home>" >&2; exit 1; }
  REMOTE_CODE="${REMOTE_REPO:-$repo_name}/${repo_path%/}"
fi

runs_dir="$(awk -F': *' '/^runs_dir:/ {sub(/ +#.*/, "", $2); print $2; exit}' "$CONFIG")"
case "$runs_dir" in
  /*)  REMOTE_RUNS="$runs_dir" ;;
  ~/*) REMOTE_RUNS="${runs_dir#\~/}" ;;          # relative to the robot user's home
  *)   REMOTE_RUNS="$REMOTE_CODE/$runs_dir" ;;   # relative to the package folder
esac

if [[ -z "${ROBOT_HOST:-}" ]]; then
  case "$TARGET" in
    g2)    ENTRY="G2_Agi" ;;
    anvil) ENTRY="Anvil_calib" ;;
    *) echo "set ROBOT_HOST=user@ip for robot $TARGET" >&2; exit 1 ;;
  esac
  if [[ -z "${INSTANCES:-}" ]]; then
    d="$HERE"
    while [[ "$d" != "/" && ! -f "$d/instances.txt" ]]; do d="$(dirname "$d")"; done
    INSTANCES="$d/instances.txt"
  fi
  [[ -f "$INSTANCES" ]] || { echo "no ROBOT_HOST and no instances.txt found" >&2; exit 1; }
  line="$(awk -v e="$ENTRY" '$0 ~ "^# *"e {getline; print; exit}' "$INSTANCES")"
  [[ -n "$line" ]] || { echo "no '# $ENTRY' entry in $INSTANCES" >&2; exit 1; }
  ROBOT_HOST="$(sed -E 's/^ *ssh +//; s/ *\|.*$//' <<<"$line")"
  ROBOT_PASS="${ROBOT_PASS:-$(sed -E 's/^[^|]*\| *//; s/ *$//' <<<"$line")}"
fi

SSH=(ssh -o ConnectTimeout=15)
if [[ -n "${ROBOT_PASS:-}" ]] && command -v sshpass >/dev/null; then
  export SSHPASS="$ROBOT_PASS"
  SSH=(sshpass -e "${SSH[@]}")
fi
RSYNC=(rsync -az --rsh="${SSH[*]}")
# Excluded paths are never deleted by --delete. GDK writes glog files (*.INFO*) on the G2.
CODE_FILTER=(--exclude runs/ --exclude logs/ --exclude __pycache__/ --exclude '*.pyc'
             --exclude .DS_Store --exclude '*.INFO*')

case "${1:-}" in
  push)
    "${SSH[@]}" "$ROBOT_HOST" "mkdir -p ~/$REMOTE_CODE"
    "${RSYNC[@]}" --delete "${CODE_FILTER[@]}" "$HERE/" "$ROBOT_HOST:$REMOTE_CODE/"
    echo "pushed -> $ROBOT_HOST:~/$REMOTE_CODE"
    ;;
  status)
    "${RSYNC[@]}" --delete --dry-run --itemize-changes "${CODE_FILTER[@]}" "$HERE/" "$ROBOT_HOST:$REMOTE_CODE/"
    ;;
  pull)
    run="${2:?usage: ./sync.sh pull <run>}"
    mkdir -p "$HERE/runs"
    "${RSYNC[@]}" "$ROBOT_HOST:$REMOTE_RUNS/$run" "$HERE/runs/"
    echo "pulled -> $HERE/runs/$run"
    ;;
  view)
    port="${WEB_PORT:-9701}"
    echo "tunnel localhost:$port -> $ROBOT_HOST:127.0.0.1:$port  (Ctrl-C to close)"
    (sleep 2; open "http://localhost:$port/" 2>/dev/null || xdg-open "http://localhost:$port/" 2>/dev/null || true) &
    "${SSH[@]}" -N -o ExitOnForwardFailure=yes -L "$port:127.0.0.1:$port" "$ROBOT_HOST"
    ;;
  *)
    sed -n '2,13p' "$0"; exit 1 ;;
esac
