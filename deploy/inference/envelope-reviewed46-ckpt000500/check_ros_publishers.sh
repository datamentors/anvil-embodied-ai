#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 \
  || ( "$1" != "0" && "$1" != "1" ) \
  || ( "${2:-0}" != "0" && "${2:-0}" != "1" ) ]]; then
  echo "Usage: $0 EXPECTED_LIVE_COMMAND_PUBLISHERS(0|1) [EXPECTED_DEBUG_COMMAND_PUBLISHERS(0|1)]" >&2
  exit 2
fi
expected_live_publishers="$1"
expected_debug_publishers="${2:-0}"

sensor_topics=(
  /joint_states
  /cam_chest/image_raw/compressed
  /cam_wrist_l/image_raw/compressed
  /cam_wrist_r/image_raw/compressed
)
live_command_topics=(
  /follower_l_forward_position_controller/commands
  /follower_r_forward_position_controller/commands
)
debug_command_topics=(
  /debug/follower_l_forward_position_controller/commands
  /debug/follower_r_forward_position_controller/commands
)

declare -A expected_sensor_publishers=(
  [/joint_states]=1
  [/cam_chest/image_raw/compressed]=2
  [/cam_wrist_l/image_raw/compressed]=2
  [/cam_wrist_r/image_raw/compressed]=2
)
declare -A expected_sensor_publisher_nodes=(
  [/joint_states]="/joint_state_broadcaster"
  [/cam_chest/image_raw/compressed]="/cam_chest/cam_chest,/cam_chest/cam_chest"
  [/cam_wrist_l/image_raw/compressed]="/cam_wrist_l/cam_wrist_l,/cam_wrist_l/cam_wrist_l"
  [/cam_wrist_r/image_raw/compressed]="/cam_wrist_r/cam_wrist_r,/cam_wrist_r/cam_wrist_r"
)
declare -A required_live_subscriber_nodes=(
  [/follower_l_forward_position_controller/commands]=\
"/follower_l_forward_position_controller"
  [/follower_r_forward_position_controller/commands]=\
"/follower_r_forward_position_controller"
)

# Populated by query_topic. Node lists are sorted canonical fully-qualified
# names and retain duplicates, so the two camera endpoints from the same node
# are both accounted for. GIDs are intentionally not part of the contract.
snapshot_publishers=""
snapshot_subscriptions=""
snapshot_publisher_nodes=""
snapshot_subscription_nodes=""
contract_error=""

query_topic() {
  local topic="$1" output snapshot extra

  if ! output="$(ros2 topic info --no-daemon --spin-time 3 --verbose "${topic}" 2>&1)"; then
    if grep -Fqi 'unknown topic' <<<"${output}"; then
      snapshot_publishers=0
      snapshot_subscriptions=0
      snapshot_publisher_nodes=""
      snapshot_subscription_nodes=""
      return 0
    fi
    echo "ERROR: ROS verbose graph query failed for ${topic}: ${output}" >&2
    return 2
  fi

  if ! snapshot="$(python3 -c '
import re
import sys

topic = sys.argv[1]
payload = sys.stdin.read()

def fail(message):
    print(f"ERROR: {message} for {topic}", file=sys.stderr)
    raise SystemExit(1)

publisher_counts = re.findall(r"^Publisher count:\s*(\d+)\s*$", payload, re.MULTILINE)
subscription_counts = re.findall(r"^Subscription count:\s*(\d+)\s*$", payload, re.MULTILINE)
if len(publisher_counts) != 1 or len(subscription_counts) != 1:
    fail("could not parse unique publisher/subscription counts")

endpoint_nodes = {"PUBLISHER": [], "SUBSCRIPTION": []}
node_name = None
node_namespace = None
for raw_line in payload.splitlines():
    line = raw_line.strip()
    if line.startswith("Node name:"):
        node_name = line.split(":", 1)[1].strip()
        node_namespace = None
    elif line.startswith("Node namespace:"):
        node_namespace = line.split(":", 1)[1].strip()
    elif line.startswith("Endpoint type:"):
        endpoint_type = line.split(":", 1)[1].strip().upper()
        if endpoint_type not in endpoint_nodes:
            fail(f"unexpected endpoint type {endpoint_type!r}")
        if not node_name or not node_namespace:
            fail("endpoint is missing node name or namespace")
        parts = [node_namespace.strip("/"), node_name.strip("/")]
        canonical_name = "/" + "/".join(part for part in parts if part)
        endpoint_nodes[endpoint_type].append(canonical_name)
        node_name = None
        node_namespace = None

publisher_count = int(publisher_counts[0])
subscription_count = int(subscription_counts[0])
if len(endpoint_nodes["PUBLISHER"]) != publisher_count:
    fail("verbose publisher endpoint records do not match Publisher count")
if len(endpoint_nodes["SUBSCRIPTION"]) != subscription_count:
    fail("verbose subscription endpoint records do not match Subscription count")

publisher_nodes = ",".join(sorted(endpoint_nodes["PUBLISHER"]))
subscription_nodes = ",".join(sorted(endpoint_nodes["SUBSCRIPTION"]))
print(f"{publisher_count}|{subscription_count}|{publisher_nodes}|{subscription_nodes}")
' "${topic}" <<<"${output}")"; then
    echo "ERROR: could not parse verbose ROS graph for ${topic}" >&2
    return 2
  fi

  IFS='|' read -r \
    snapshot_publishers \
    snapshot_subscriptions \
    snapshot_publisher_nodes \
    snapshot_subscription_nodes \
    extra <<<"${snapshot}"
  if [[ -n "${extra}" \
    || ! "${snapshot_publishers}" =~ ^[0-9]+$ \
    || ! "${snapshot_subscriptions}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid parsed ROS graph snapshot for ${topic}: ${snapshot}" >&2
    return 2
  fi
}

# Use "*" for a count or node multiset that is intentionally unconstrained.
# Return 1 for a valid but unexpected graph and 2 for a query/parse failure.
check_topic_contract() {
  local topic="$1"
  local expected_publishers="$2"
  local expected_subscriptions="$3"
  local expected_publisher_nodes="$4"
  local expected_subscription_nodes="$5"
  local mismatches=()

  contract_error=""
  if ! query_topic "${topic}"; then
    return 2
  fi

  if [[ "${expected_publishers}" != "*" \
    && "${snapshot_publishers}" != "${expected_publishers}" ]]; then
    mismatches+=("publishers=${snapshot_publishers}, expected=${expected_publishers}")
  fi
  if [[ "${expected_subscriptions}" != "*" \
    && "${snapshot_subscriptions}" != "${expected_subscriptions}" ]]; then
    mismatches+=("subscriptions=${snapshot_subscriptions}, expected=${expected_subscriptions}")
  fi
  if [[ "${expected_publisher_nodes}" != "*" \
    && "${snapshot_publisher_nodes}" != "${expected_publisher_nodes}" ]]; then
    mismatches+=(
      "publisher_nodes=${snapshot_publisher_nodes:-<none>}, "\
"expected=${expected_publisher_nodes:-<none>}"
    )
  fi
  if [[ "${expected_subscription_nodes}" != "*" \
    && "${snapshot_subscription_nodes}" != "${expected_subscription_nodes}" ]]; then
    mismatches+=(
      "subscription_nodes=${snapshot_subscription_nodes:-<none>}, "\
"expected=${expected_subscription_nodes:-<none>}"
    )
  fi

  if (( ${#mismatches[@]} != 0 )); then
    contract_error="${topic}: $(IFS='; '; echo "${mismatches[*]}")"
    return 1
  fi
  contract_error=""
}

check_sensor_contract() {
  local topic="$1"
  check_topic_contract \
    "${topic}" \
    "${expected_sensor_publishers[${topic}]}" \
    "*" \
    "${expected_sensor_publisher_nodes[${topic}]}" \
    "*"
}

check_live_command_contract() {
  local topic="$1" publisher_nodes="" required_subscriber allowed_with_replay status
  if [[ "${expected_live_publishers}" == "1" ]]; then
    publisher_nodes="/lerobot_inference"
  fi
  if check_topic_contract \
    "${topic}" \
    "${expected_live_publishers}" \
    "*" \
    "${publisher_nodes}" \
    "*"; then
    :
  else
    status=$?
    return "${status}"
  fi

  # replay_buffer is a passive recorder, not a controller authority endpoint.
  # It may be absent after a deliberate stop or a process failure. Require the
  # real controller exactly once and permit at most that one known recorder.
  required_subscriber="${required_live_subscriber_nodes[${topic}]}"
  allowed_with_replay="${required_subscriber},/replay_buffer"
  if [[ "${snapshot_subscriptions}" == "1" \
      && "${snapshot_subscription_nodes}" == "${required_subscriber}" ]] \
    || [[ "${snapshot_subscriptions}" == "2" \
      && "${snapshot_subscription_nodes}" == "${allowed_with_replay}" ]]; then
    contract_error=""
    return 0
  fi

  contract_error="${topic}: subscription_nodes="\
"${snapshot_subscription_nodes:-<none>}, expected=${required_subscriber}"\
"[,+optional /replay_buffer]"
  return 1
}

check_debug_command_contract() {
  local topic="$1" publisher_nodes="" status
  # The vendor replay buffer passively records debug commands. It has no
  # controller authority, but it is the only permitted debug subscriber.
  if [[ "${expected_debug_publishers}" == "1" ]]; then
    publisher_nodes="/lerobot_inference"
  fi
  if check_topic_contract \
    "${topic}" \
    "${expected_debug_publishers}" \
    "*" \
    "${publisher_nodes}" \
    "*"; then
    :
  else
    status=$?
    return "${status}"
  fi

  # Debug topics never reach a controller. The same passive replay recorder is
  # the only optional subscriber; any other endpoint is an authority surprise.
  if [[ "${snapshot_subscriptions}" == "0" \
      && -z "${snapshot_subscription_nodes}" ]] \
    || [[ "${snapshot_subscriptions}" == "1" \
      && "${snapshot_subscription_nodes}" == "/replay_buffer" ]]; then
    contract_error=""
    return 0
  fi

  contract_error="${topic}: subscription_nodes="\
"${snapshot_subscription_nodes:-<none>}, expected=<none> or /replay_buffer"
  return 1
}

deadline=$((SECONDS + 60))
while true; do
  sensors_ready=true
  sensor_errors=()
  for topic in "${sensor_topics[@]}"; do
    if check_sensor_contract "${topic}"; then
      continue
    else
      status=$?
    fi
    if (( status == 2 )); then
      exit 1
    fi
    sensors_ready=false
    sensor_errors+=("${contract_error}")
  done
  if [[ "${sensors_ready}" == true ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "ERROR: DDS discovery did not reach the required real-sensor endpoint contract:" >&2
    printf '  %s\n' "${sensor_errors[@]}" >&2
    exit 1
  fi
  sleep 1
done

# Require the complete graph contract three consecutive times after discovery.
# A transient snapshot is not sufficient for a real-robot authority gate.
for sample in 1 2 3; do
  for topic in "${sensor_topics[@]}"; do
    if ! check_sensor_contract "${topic}"; then
      echo "ERROR: ${contract_error:-sensor graph query failed} (sample ${sample}/3)" >&2
      exit 1
    fi
  done
  for topic in "${live_command_topics[@]}"; do
    if ! check_live_command_contract "${topic}"; then
      echo "ERROR: ${contract_error:-live command graph query failed} (sample ${sample}/3)" >&2
      exit 1
    fi
  done
  for topic in "${debug_command_topics[@]}"; do
    if ! check_debug_command_contract "${topic}"; then
      echo "ERROR: ${contract_error:-debug command graph query failed} (sample ${sample}/3)" >&2
      exit 1
    fi
  done
  if (( sample < 3 )); then
    sleep 1
  fi
done

printf 'DDS_AUTHORITY_PASS live_publishers=%s ' \
  "${expected_live_publishers}"
printf '%s ' 'live_subscribers=controller[+optional_replay_buffer]'
printf 'debug_publishers=%s ' "${expected_debug_publishers}"
printf '%s ' 'debug_subscribers=optional_replay_buffer'
printf '%s\n' 'sensors=4 samples=3 identities=verified'
