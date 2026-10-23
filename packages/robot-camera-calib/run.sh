#!/usr/bin/env bash
# Run the tool with dependencies from uv (nothing installed system-wide).
# Use it on any host without OpenCV >= 4.7 / PyYAML / websockets, e.g. a ROS robot's host.
#
#   ./run.sh capture --robot anvil --run anvil01 --board <spec> --web
#   ./run.sh solve <runs_dir>/anvil01
#   ./run.sh selftest
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
[[ -x "$UV" ]] || { echo "uv not found (looked on PATH and ~/.local/bin/uv)" >&2; exit 1; }

cmd="${1:-}"; shift || true
case "$cmd" in
  capture|solve|selftest)
    exec "$UV" run -q --no-project --python 3.12 \
      --with "opencv-contrib-python-headless==4.13.*" --with numpy --with pyyaml --with "websockets>=12" \
      python -m "robot_camera_calib.$cmd" "$@" ;;
  *) sed -n '2,7p' "$0"; exit 1 ;;
esac
