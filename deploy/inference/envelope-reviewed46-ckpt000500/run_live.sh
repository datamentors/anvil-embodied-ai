#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The confirmation gate lives in the internal runner too, so invoking that
# helper directly cannot bypass real-robot authorization.
exec "${DEPLOY_DIR}/_run_mode.sh" live
