#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 echo|shadow|shadow_quiet|shadow_joint_worker|shadow_joint_worker_monitor|live" >&2
  exit 2
fi

# Capture this before runner output is intentionally redirected through tee.
# After that redirection stdout is a pipe even when the operator launched us
# from a genuine terminal.
interactive_terminal=false
if [[ -t 0 && -t 1 ]]; then
  interactive_terminal=true
fi

MODE="$1"
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${RUNTIME_ENV_FILE:-${DEPLOY_DIR}/runtime.env}"
COMPOSE_FILE="${DEPLOY_DIR}/docker-compose.gpu.yml"
PROJECT_NAME="envelope-reviewed46-ckpt000500"
shadow_mode=false

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
export JOINT_STATE_WORKER=false

case "${MODE}" in
  echo)
    export CONFIG_FILE="${DEPLOY_DIR}/inference_envelope_ckpt000500_shadow.yaml"
    export ECHO_TOPIC_ONLY=true
    export DEBUG=false
    export MONITOR_ENABLE=false
    services=(inference)
    success_markers=("Mode:       Monitor Only" "Starting inference loop")
    ;;
  shadow)
    shadow_mode=true
    export CONFIG_FILE="${DEPLOY_DIR}/inference_envelope_ckpt000500_shadow.yaml"
    export ECHO_TOPIC_ONLY=false
    export DEBUG=true
    export MONITOR_ENABLE=true
    services=(inference inference-monitor)
    success_markers=(
      "Verified checkpoint manifest (9 artifacts)"
      "Model loaded successfully on cuda"
      "Loaded preprocessor pipeline from checkpoint"
      "Loaded postprocessor pipeline from checkpoint"
      "RTC enabled for pi05"
      "Starting inference loop"
      "[RTC] POLICY_READY"
    )
    ;;
  shadow_quiet)
    # Diagnostic A/B: preserve the full model, RTC queue consumption and
    # isolated debug publishers, but remove per-action logging and monitor
    # publishers/CSV so their post-readiness cost can be measured directly.
    shadow_mode=true
    export CONFIG_FILE="${DEPLOY_DIR}/inference_envelope_ckpt000500_shadow.yaml"
    export ECHO_TOPIC_ONLY=false
    export DEBUG=false
    export MONITOR_ENABLE=false
    services=(inference)
    success_markers=(
      "Verified checkpoint manifest (9 artifacts)"
      "Model loaded successfully on cuda"
      "Loaded preprocessor pipeline from checkpoint"
      "Loaded postprocessor pipeline from checkpoint"
      "RTC enabled for pi05"
      "Starting inference loop"
      "[RTC] POLICY_READY"
    )
    ;;
  shadow_joint_worker)
    # Same quiet shadow contract, with the 500 Hz JointState subscription
    # isolated in a spawned ROS2 worker. MultiProcessStrategy rejects this
    # option at startup unless every configured command topic is under /debug/.
    shadow_mode=true
    export CONFIG_FILE="${DEPLOY_DIR}/inference_envelope_ckpt000500_shadow.yaml"
    export ECHO_TOPIC_ONLY=false
    export DEBUG=false
    export MONITOR_ENABLE=false
    export JOINT_STATE_WORKER=true
    services=(inference)
    success_markers=(
      "Verified checkpoint manifest (9 artifacts)"
      "Started worker: /joint_states -> joint states"
      "joint_state_worker=True"
      "Model loaded successfully on cuda"
      "Loaded preprocessor pipeline from checkpoint"
      "Loaded postprocessor pipeline from checkpoint"
      "RTC enabled for pi05"
      "Starting inference loop"
      "[RTC] POLICY_READY"
    )
    ;;
  shadow_joint_worker_monitor)
    # Five-minute acceptance diagnostic: isolate the 500 Hz JointState reader
    # while retaining the monitor CSV needed to audit all 16 outputs.
    shadow_mode=true
    export CONFIG_FILE="${DEPLOY_DIR}/inference_envelope_ckpt000500_shadow.yaml"
    export ECHO_TOPIC_ONLY=false
    export DEBUG=false
    export MONITOR_ENABLE=true
    export JOINT_STATE_WORKER=true
    services=(inference inference-monitor)
    success_markers=(
      "Verified checkpoint manifest (9 artifacts)"
      "Started worker: /joint_states -> joint states"
      "joint_state_worker=True"
      "Model loaded successfully on cuda"
      "Loaded preprocessor pipeline from checkpoint"
      "Loaded postprocessor pipeline from checkpoint"
      "RTC enabled for pi05"
      "Starting inference loop"
      "[RTC] POLICY_READY"
    )
    ;;
  live)
    export CONFIG_FILE="${DEPLOY_DIR}/inference_envelope_ckpt000500_live.yaml"
    export ECHO_TOPIC_ONLY=false
    export DEBUG=false
    export MONITOR_ENABLE=true
    export JOINT_STATE_WORKER=true
    services=(inference inference-monitor)
    success_markers=(
      "Verified checkpoint manifest (9 artifacts)"
      "Started worker: /joint_states -> joint states"
      "joint_state_worker=True"
      "Model loaded successfully on cuda"
      "Loaded preprocessor pipeline from checkpoint"
      "Loaded postprocessor pipeline from checkpoint"
      "RTC enabled for pi05"
      "Starting inference loop"
      "[RTC] POLICY_READY"
    )
    ;;
  *)
    echo "ERROR: unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac

# The first shadow captured and reviewed all three camera aliases. Subsequent
# performance runs keep debug metrics but avoid PNG I/O in camera callbacks.
export CAPTURE_DEBUG_IMAGES=false

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
export DEBUG_IMAGE_DIR="${DEPLOY_DIR}/outputs/${MODE}-${timestamp}/debug_images"
export MONITOR_OUTPUT_DIR="${DEPLOY_DIR}/outputs/${MODE}-${timestamp}/monitor"
telemetry_dir="${DEPLOY_DIR}/outputs/${MODE}-${timestamp}/telemetry"
mkdir -p \
  "${DEBUG_IMAGE_DIR}" \
  "${MONITOR_OUTPUT_DIR}" \
  "${telemetry_dir}" \
  "${DEPLOY_DIR}/logs"

# Keep runner/Compose-driver output separate from the canonical container-log
# snapshot. The latter is atomically refreshed, so repeated safety checks never
# duplicate container output or overwrite the runner's own failure evidence.
log_file="${DEPLOY_DIR}/logs/${MODE}-${timestamp}.log"
driver_log_file="${DEPLOY_DIR}/logs/${MODE}-${timestamp}.driver.log"
supervision_event_file="${DEPLOY_DIR}/logs/${MODE}-${timestamp}.supervision"
: >"${log_file}"
: >"${driver_log_file}"
: >"${supervision_event_file}"
exec > >(tee -a "${driver_log_file}") 2>&1

echo "[evidence] container log: ${log_file}"
echo "[evidence] runner transcript: ${driver_log_file}"

"${DEPLOY_DIR}/preflight.sh"

compose=(
  docker compose
  --project-name "${PROJECT_NAME}"
  --env-file "${ENV_FILE}"
  -f "${COMPOSE_FILE}"
  --profile tools
)
if [[ "${MONITOR_ENABLE}" == "true" ]]; then
  compose+=(--profile monitor)
fi

if "${compose[@]}" ps --status running --quiet inference | grep -q .; then
  echo "REFUSED: this isolated inference deployment is already running." >&2
  echo "Stop the current echo/shadow/live session (or run ./stop.sh) before changing mode." >&2
  exit 1
fi

require_graph_contract() {
  local expected_live="$1"
  local expected_debug="$2"
  local phase="$3"
  echo "[authority] ${phase}: checking sensors and command endpoint ownership"
  require_supervision_ok
  if ! run_interruptible "${compose[@]}" run --rm --no-deps dds-check \
    bash /workspace/check_ros_publishers.sh "${expected_live}" "${expected_debug}"; then
    fail_no_go "DDS authority gate failed during: ${phase}"
  fi
  require_supervision_ok
}

require_robot_home() {
  echo "[home] requesting the existing arms reset and verifying the measured start pose"
  require_supervision_ok
  if ! run_interruptible "${compose[@]}" run --rm --no-deps dds-check \
    python3 /workspace/prepare_robot_home.py \
      --contract /workspace/configs/lerobot_control/robot_home_contract.json; then
    fail_no_go "robot homing or start-pose verification failed"
  fi
  require_supervision_ok
}

fatal_log_pattern='Returning model without loading pretrained weights|Could not load state dict|Missing keys when loading state dict|Unexpected keys when loading state dict|Failed to load processor pipelines|Built pi05 processor from policy factory|No processor pipelines found|Traceback \(most recent call last\)|\[WATCHDOG\] LATCHED'

compose_started=false
supervisor_pid=""
gate_pid=""
telemetry_pids=()
runner_pid="$$"

run_interruptible() {
  # Bash defers signal traps while a foreground external command owns the
  # shell. Run long Compose operations as children so Ctrl-C/HUP/TERM reaches
  # handle_signal immediately and cleanup can tear down the isolated project.
  local status

  "$@" &
  gate_pid="$!"
  set +e
  wait "${gate_pid}"
  status="$?"
  set -e
  gate_pid=""
  return "${status}"
}

refresh_logs() {
  local refresh_tmp="${log_file}.tmp.${BASHPID}"

  if [[ "${compose_started}" != "true" ]]; then
    return 0
  fi
  if "${compose[@]}" logs --no-color --timestamps >"${refresh_tmp}" 2>&1; then
    mv -f "${refresh_tmp}" "${log_file}"
    return 0
  fi

  rm -f "${refresh_tmp}"
  return 1
}

fatal_reason_from_log() {
  if grep -Fq '[WATCHDOG] LATCHED' "${log_file}"; then
    printf '%s\n' 'watchdog LATCHED condition detected in the container log'
  elif grep -Fq 'Traceback (most recent call last)' "${log_file}"; then
    printf '%s\n' 'Python Traceback detected in the container log'
  elif grep -Eq "${fatal_log_pattern}" "${log_file}"; then
    printf '%s\n' 'fail-open model/processor fallback detected in the container log'
  fi
}

record_supervision_failure() {
  local reason="$1"

  if [[ ! -s "${supervision_event_file}" ]]; then
    printf '%s\n' "${reason}" >"${supervision_event_file}"
  fi
}

fail_no_go() {
  local reason="$1"
  record_supervision_failure "${reason}"
  # A concurrent supervisor HUP must not print the same terminal condition a
  # second time while this explicit failure path is exiting.
  trap '' HUP
  echo "NO-GO: $(<"${supervision_event_file}")" >&2
  exit 1
}

require_supervision_ok() {
  if [[ -s "${supervision_event_file}" ]]; then
    fail_no_go "$(<"${supervision_event_file}")"
  fi
  if [[ -n "${supervisor_pid}" ]] && ! kill -0 "${supervisor_pid}" 2>/dev/null; then
    fail_no_go "runtime safety supervisor stopped unexpectedly"
  fi
}

require_no_fatal_log() {
  local reason

  if ! refresh_logs; then
    fail_no_go "could not refresh the container log"
  fi
  reason="$(fatal_reason_from_log)"
  if [[ -n "${reason}" ]]; then
    fail_no_go "${reason}"
  fi
  require_supervision_ok
}

require_services_running() {
  local service running
  for service in "${services[@]}"; do
    running="$("${compose[@]}" ps --status running --quiet "${service}")"
    if [[ -z "${running}" ]]; then
      refresh_logs || true
      fail_no_go "required service stopped: ${service}"
    fi
  done
}

stop_inference_container_now() {
  # Command authority must disappear before monitors or tooling. Compose may
  # otherwise spend its graceful timeout stopping the monitor first while the
  # inference node continues publishing to the robot.
  local container_id
  while IFS= read -r container_id; do
    if [[ -n "${container_id}" ]]; then
      docker stop --time 2 "${container_id}" >/dev/null 2>&1 || true
    fi
  done < <(
    docker ps --quiet \
      --filter "label=com.docker.compose.project=${PROJECT_NAME}" \
      --filter "label=com.docker.compose.service=inference"
  )
}

stop_project_containers_now() {
  # A signal trap in the parent shell is deferred while `docker compose run`
  # owns the foreground. Stop only this uniquely labelled project first so a
  # fatal watchdog event cannot leave inference running until a long DDS graph
  # query returns.
  local container_id
  stop_inference_container_now
  while IFS= read -r container_id; do
    if [[ -n "${container_id}" ]]; then
      docker stop --time 2 "${container_id}" >/dev/null 2>&1 || true
    fi
  done < <(
    docker ps --quiet \
      --filter "label=com.docker.compose.project=${PROJECT_NAME}"
  )
}

start_supervision() {
  # Populate the log immediately after compose up. From this point through all
  # post-start gates and the attended runtime, the supervisor refreshes the
  # snapshot and tears the run down on a fatal log condition or service death.
  if ! refresh_logs; then
    fail_no_go "could not capture the initial container log"
  fi
  (
    set +e
    while true; do
      if ! refresh_logs; then
        record_supervision_failure "could not refresh the container log"
        stop_project_containers_now
        kill -HUP "${runner_pid}" 2>/dev/null
        exit 1
      fi

      reason="$(fatal_reason_from_log)"
      if [[ -n "${reason}" ]]; then
        record_supervision_failure "${reason}"
        stop_project_containers_now
        kill -HUP "${runner_pid}" 2>/dev/null
        exit 1
      fi

      for service in "${services[@]}"; do
        if [[ -z "$("${compose[@]}" ps --status running --quiet "${service}")" ]]; then
          refresh_logs || true
          record_supervision_failure "required service stopped: ${service}"
          stop_project_containers_now
          kill -HUP "${runner_pid}" 2>/dev/null
          exit 1
        fi
      done
      sleep 1
    done
  ) &
  supervisor_pid="$!"
}

start_runtime_telemetry() {
  # Diagnostics are evidence only: they never influence a watchdog decision or
  # readiness gate. Keep each collector independent so a missing optional host
  # utility cannot prevent a fail-closed shadow run.
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s pucvmet -d 1 -o DT \
      >"${telemetry_dir}/nvidia-dmon.log" 2>&1 &
    telemetry_pids+=("$!")
  fi
  if command -v mpstat >/dev/null 2>&1; then
    mpstat -P ALL 1 >"${telemetry_dir}/mpstat.log" 2>&1 &
    telemetry_pids+=("$!")
  fi
  (
    while true; do
      date -u +'%Y-%m-%dT%H:%M:%SZ'
      "${compose[@]}" ps --format json inference || exit 1
      container_id="$("${compose[@]}" ps --quiet inference)"
      if [[ -n "${container_id}" ]]; then
        docker stats --no-stream --format \
          'cpu={{.CPUPerc}} mem={{.MemUsage}} net={{.NetIO}} block={{.BlockIO}} pids={{.PIDs}}' \
          "${container_id}" || exit 1
        docker top "${container_id}" -eo pid,ppid,nlwp,psr,pcpu,pmem,rss,stat,comm \
          || exit 1
      fi
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi \
          --query-compute-apps=pid,process_name,used_memory \
          --format=csv,noheader || exit 1
      fi
      sleep 5
    done
  ) >"${telemetry_dir}/container-runtime.log" 2>&1 &
  telemetry_pids+=("$!")
  echo "[evidence] runtime telemetry: ${telemetry_dir}"
}

cleanup() {
  local status="$?"
  trap - EXIT HUP INT TERM

  set +e
  stop_inference_container_now
  if [[ -n "${gate_pid}" ]]; then
    kill -TERM "${gate_pid}" >/dev/null 2>&1 || true
    wait "${gate_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${supervisor_pid}" ]]; then
    kill "${supervisor_pid}" >/dev/null 2>&1 || true
    wait "${supervisor_pid}" >/dev/null 2>&1 || true
  fi
  for telemetry_pid in "${telemetry_pids[@]}"; do
    kill "${telemetry_pid}" >/dev/null 2>&1 || true
  done
  for telemetry_pid in "${telemetry_pids[@]}"; do
    wait "${telemetry_pid}" >/dev/null 2>&1 || true
  done
  # Docker removes the only copy of a container's stdout on `down`. Always
  # refresh the canonical snapshot first, including on signals and failures.
  if [[ "${compose_started}" == "true" ]]; then
    if ! refresh_logs; then
      echo "WARNING: final container-log refresh failed; preserving the previous snapshot." >&2
    fi
  fi
  echo "[evidence] final container log: ${log_file}"
  "${compose[@]}" down --remove-orphans || true
  echo "[evidence] final runner transcript: ${driver_log_file}"
  exit "${status}"
}
handle_signal() {
  local signal="$1"
  if [[ -s "${supervision_event_file}" ]]; then
    echo "NO-GO: $(<"${supervision_event_file}")" >&2
  else
    echo "[${MODE}] interrupted by ${signal}; stopping the isolated deployment." >&2
  fi
  exit 1
}
trap cleanup EXIT
trap 'handle_signal HUP' HUP
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

# Build/check the isolated image before any live-robot confirmation. This keeps
# the final confirmation immediately adjacent to creating command publishers.
if ! run_interruptible "${compose[@]}" build inference; then
  fail_no_go "docker compose build failed"
fi

if [[ "${shadow_mode}" == "true" ]]; then
  # Shadow must begin from a graph with no live publisher and no pre-existing
  # debug endpoint. The checker also binds the real sensors and live command
  # subscribers to the endpoint identities observed on this workcell.
  require_graph_contract 0 0 "before shadow startup"
elif [[ "${MODE}" == "live" ]]; then
  # The graph must have no existing control authority before we create live
  # publishers. This catches unknown publishers, not only named teleop stacks.
  require_graph_contract 0 0 "before live startup"
  ENV_CONFIRM="RUN_CKPT000500_ON_REAL_ROBOT"
  TYPED_CONFIRM="HOME AND RUN CKPT000500 LIVE"
  if [[ "${LIVE_ROBOT_CONFIRM:-}" != "${ENV_CONFIRM}" ]]; then
    echo "REFUSED: export LIVE_ROBOT_CONFIRM=${ENV_CONFIRM}" >&2
    exit 1
  fi
  if [[ "${interactive_terminal}" != "true" || ! -e /dev/tty ]]; then
    echo "REFUSED: live inference requires an interactive terminal." >&2
    exit 1
  fi
  echo "WARNING: this first HOMES both real arms, then publishes absolute joint targets."
  echo "Required: operator present, E-stop tested, cables and workspace clear, shadow review approved."
  read -r -p "Type exactly '${TYPED_CONFIRM}': " answer </dev/tty
  if [[ "${answer}" != "${TYPED_CONFIRM}" ]]; then
    echo "REFUSED: confirmation did not match." >&2
    exit 1
  fi
  require_robot_home
  # Homing switches controllers internally. Re-sample the complete graph after
  # it finishes so inference cannot start if authority or sensor ownership
  # changed during that physical movement.
  require_graph_contract 0 0 "after homing before live startup"
fi

# Mark the Compose attempt before invoking the driver so even a partial startup
# failure gets a final log snapshot before cleanup removes its containers.
compose_started=true
if ! "${compose[@]}" up --no-build --detach "${services[@]}"; then
  fail_no_go "docker compose up failed"
fi
start_runtime_telemetry
start_supervision

deadline=$((SECONDS + 300))
for marker in "${success_markers[@]}"; do
  while true; do
    require_no_fatal_log

    if grep -Fq "${marker}" "${log_file}"; then
      break
    fi
    if (( SECONDS >= deadline )); then
      fail_no_go "startup timed out waiting for: ${marker}"
    fi
    require_services_running
    sleep 1
  done
done

if [[ "${MODE}" == "live" ]]; then
  # Exactly one publisher per arm must remain after startup: this inference
  # node. Any ambiguity tears the isolated deployment down through EXIT trap.
  require_graph_contract 1 0 "after live startup"
elif [[ "${shadow_mode}" == "true" ]]; then
  # A successful policy-ready marker is not enough: prove that the process has
  # authority only over the isolated debug endpoints and none over the real
  # controllers before declaring the shadow session ready for observation.
  require_graph_contract 0 1 "after shadow policy readiness"
  echo "[shadow] proving finite 8-D action flow on both isolated debug topics"
  require_supervision_ok
  if ! run_interruptible "${compose[@]}" run --rm --no-deps dds-check \
    python3 /workspace/check_shadow_flow.py; then
    fail_no_go "shadow action-flow gate failed"
  fi
  require_supervision_ok
  # The passive flow checker must disappear cleanly. replay_buffer may remain
  # as the only approved debug subscriber, or the topic may have none.
  require_graph_contract 0 1 "after shadow flow check"
fi

# The graph queries above can take tens of seconds. Re-check both process
# liveness and the safety log after them so a latch in that window can never be
# reported as a successful startup.
require_services_running
require_no_fatal_log

echo "[${MODE}] Startup contract passed. Logs: ${log_file}"
if [[ "${MODE}" == "echo" ]]; then
  echo "[echo] Read-only DDS observation; this mode has no command publishers."
elif [[ "${shadow_mode}" == "true" ]]; then
  echo "[shadow] Commands are isolated under /debug/*; no live controller topics are used."
else
  echo "[live] LIVE controller topics are active. Keep the operator and E-stop ready."
fi
echo "Press Ctrl-C to stop; the isolated containers will be removed automatically."

# Supervision already started immediately after `compose up` and remained
# active throughout every gate. Keep the attended driver alive without opening
# a second Compose log stream; the atomically refreshed snapshot is the single
# non-duplicated source of container stdout.
while true; do
  require_supervision_ok
  sleep 1
done
