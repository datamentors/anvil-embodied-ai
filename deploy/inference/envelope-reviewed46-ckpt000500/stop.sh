#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_FILE="${DEPLOY_DIR}/inference_envelope_ckpt000500_shadow.yaml"
compose=(docker compose \
  --project-name envelope-reviewed46-ckpt000500 \
  --env-file "${RUNTIME_ENV_FILE:-${DEPLOY_DIR}/runtime.env}" \
  -f "${DEPLOY_DIR}/docker-compose.gpu.yml" \
  --profile monitor)

# Remove real command authority first; monitor shutdown must never extend the
# time that the inference node can keep publishing.
"${compose[@]}" stop --timeout 2 inference || true
"${compose[@]}" down --remove-orphans
