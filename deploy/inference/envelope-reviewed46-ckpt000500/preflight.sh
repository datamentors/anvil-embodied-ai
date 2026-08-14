#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${RUNTIME_ENV_FILE:-${DEPLOY_DIR}/runtime.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: runtime environment not found: ${ENV_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${HF_CACHE:?HF_CACHE is required}"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is required}"
: "${CONTROL_FREQ:?CONTROL_FREQ is required}"
: "${INFERENCE_CPU_THREADS:?INFERENCE_CPU_THREADS is required}"
: "${DDS_IFACE:?DDS_IFACE is required}"
: "${DDS_LOCAL_IP:?DDS_LOCAL_IP is required}"
: "${DDS_PEER_IP:?DDS_PEER_IP is required}"

if [[ "${ROS_DOMAIN_ID}" != "204" ]]; then
  echo "ERROR: ROS_DOMAIN_ID=${ROS_DOMAIN_ID}; expected the existing two-PC domain 204" >&2
  exit 1
fi
if [[ "${CONTROL_FREQ}" != "30" && "${CONTROL_FREQ}" != "30.0" ]]; then
  echo "ERROR: CONTROL_FREQ=${CONTROL_FREQ}; checkpoint deployment contract is 30 Hz" >&2
  exit 1
fi
if [[ "${INFERENCE_CPU_THREADS}" != "4" ]]; then
  echo "ERROR: INFERENCE_CPU_THREADS=${INFERENCE_CPU_THREADS}; reviewed shadow A/B contract is 4" >&2
  exit 1
fi

PRETRAINED="${MODEL_PATH}/pretrained_model"
if [[ ! -d "${PRETRAINED}" ]]; then
  echo "ERROR: expected checkpoint step directory with pretrained_model/: ${MODEL_PATH}" >&2
  exit 1
fi

manifest_target="${PRETRAINED}/checkpoint_manifest.sha256"
if [[ ! -f "${manifest_target}" ]]; then
  echo "ERROR: fail-closed runtime manifest missing: ${manifest_target}" >&2
  echo "Copy deploy/checkpoint_manifest.sha256 there after the model transfer." >&2
  exit 1
fi
if ! cmp -s "${DEPLOY_DIR}/checkpoint_manifest.sha256" "${manifest_target}"; then
  echo "ERROR: runtime checkpoint manifest differs from the reviewed manifest" >&2
  exit 1
fi

if [[ ! -d "${HF_CACHE}" ]]; then
  echo "ERROR: Hugging Face cache not found: ${HF_CACHE}" >&2
  exit 1
fi

for host_path in "${MODEL_PATH}" "${HF_CACHE}"; do
  if [[ "${host_path}" != /* ]]; then
    echo "ERROR: Docker bind paths must be absolute: ${host_path}" >&2
    exit 1
  fi
done

echo "[preflight] Verifying immutable checkpoint 000500..."
(
  cd "${PRETRAINED}"
  sha256sum -c "${DEPLOY_DIR}/checkpoint_manifest.sha256"
)

actual_bytes="$(find "${PRETRAINED}" -maxdepth 1 -type f \
  ! -name 'checkpoint_manifest.sha256' \
  ! -name 'SHA256SUMS.expected' \
  -printf '%s\n' | awk '{sum += $1} END {printf "%.0f", sum}')"
if [[ "${actual_bytes}" != "9354084031" ]]; then
  echo "ERROR: pretrained_model byte count ${actual_bytes}; expected 9354084031" >&2
  exit 1
fi

python3 - "${PRETRAINED}" \
  "${DEPLOY_DIR}/inference_envelope_ckpt000500_shadow.yaml" \
  "${DEPLOY_DIR}/inference_envelope_ckpt000500_live.yaml" <<'PY'
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit(f"ERROR: PyYAML is required for preflight: {exc}")

model_dir = Path(sys.argv[1])
shadow_path = Path(sys.argv[2])
live_path = Path(sys.argv[3])

expected_prompt = "Pick up the envelope and place it in the target area"
per_arm_model = ["finger_joint1", *(f"joint{i}" for i in range(1, 8))]
full_model = [f"left_{name}" for name in per_arm_model] + [
    f"right_{name}" for name in per_arm_model
]
expected_inputs = {
    "observation.images.base": ("VISUAL", [3, 480, 640]),
    "observation.images.left_wrist": ("VISUAL", [3, 480, 640]),
    "observation.images.right_wrist": ("VISUAL", [3, 480, 640]),
    "observation.state": ("STATE", [16]),
}
expected_cameras = {
    "/cam_chest/image_raw/compressed": "base",
    "/cam_wrist_l/image_raw/compressed": "left_wrist",
    "/cam_wrist_r/image_raw/compressed": "right_wrist",
}
expected_joint_limits = {
    "follower_l_joint1": [-2.3562, 2.3562],
    "follower_l_joint2": [-3.3161, 0.1745],
    "follower_l_joint3": [-1.570796, 1.570796],
    "follower_l_joint4": [0.0, 2.443461],
    "follower_l_joint5": [-1.570796, 1.570796],
    "follower_l_joint6": [-0.785398, 0.785398],
    "follower_l_joint7": [-1.570796, 1.570796],
    "follower_l_finger_joint1": [0.0, 0.05],
    "follower_r_joint1": [-2.3562, 2.3562],
    "follower_r_joint2": [-0.1745, 3.3161],
    "follower_r_joint3": [-1.570796, 1.570796],
    "follower_r_joint4": [0.0, 2.443461],
    "follower_r_joint5": [-1.570796, 1.570796],
    "follower_r_joint6": [-0.785398, 0.785398],
    "follower_r_joint7": [-1.570796, 1.570796],
    "follower_r_finger_joint1": [0.0, 0.05],
}

cfg = json.loads((model_dir / "config.json").read_text())
anvil = json.loads((model_dir / "anvil_config.json").read_text())
pre = json.loads((model_dir / "policy_preprocessor.json").read_text())
post = json.loads((model_dir / "policy_postprocessor.json").read_text())
split = json.loads((model_dir / "split_info.json").read_text())

assert cfg["type"] == "pi05"
assert cfg["n_obs_steps"] == 1
assert cfg["chunk_size"] == 50 and cfg["n_action_steps"] == 50
assert cfg["num_inference_steps"] == 10
assert cfg["dtype"] == "bfloat16"
assert cfg["use_relative_actions"] is False
assert cfg["action_feature_names"] == full_model
assert cfg["output_features"]["action"] == {"type": "ACTION", "shape": [16]}
assert set(cfg["input_features"]) == set(expected_inputs)
for key, (feature_type, shape) in expected_inputs.items():
    assert cfg["input_features"][key] == {"type": feature_type, "shape": shape}
assert cfg["normalization_mapping"] == {
    "VISUAL": "IDENTITY",
    "STATE": "QUANTILES",
    "ACTION": "QUANTILES",
}
assert anvil["action_type"] == "absolute"
assert anvil["use_delta_actions"] is False
assert anvil["task_description"] == expected_prompt
assert split["val_episodes"] == [4, 10, 29, 34, 45]
assert split["test_episodes"] == [6, 22, 25, 27, 42]

pre_steps = {step["registry_name"]: step for step in pre["steps"]}
post_steps = {step["registry_name"]: step for step in post["steps"]}
assert pre_steps["delta_actions_processor"]["config"]["enabled"] is False
assert pre_steps["normalizer_processor"]["config"]["norm_map"] == cfg["normalization_mapping"]
assert pre_steps["tokenizer_processor"]["config"]["tokenizer_name"] == "google/paligemma-3b-pt-224"
assert post_steps["unnormalizer_processor"]["config"]["norm_map"] == cfg["normalization_mapping"]
assert post_steps["absolute_actions_processor"]["config"]["enabled"] is False

for path, expected_topics, expected_horizon in (
    (
        shadow_path,
        {
            "left": "/debug/follower_l_forward_position_controller/commands",
            "right": "/debug/follower_r_forward_position_controller/commands",
        },
        35,
    ),
    (
        live_path,
        {
            "left": "/follower_l_forward_position_controller/commands",
            "right": "/follower_r_forward_position_controller/commands",
        },
        35,
    ),
):
    runtime = yaml.safe_load(path.read_text())
    rtc = runtime["inference_tuning"]["rtc"]
    assert rtc["execution_horizon"] == expected_horizon
    assert rtc["queue_trigger_threshold"] == 50
    assert rtc["max_guidance_weight"] == 10.0
    assert rtc["readiness_guided_forwards"] == 5
    assert rtc["readiness_latency_guard_steps"] == 2
    assert rtc["readiness_index_phase_tolerance_steps"] == 1
    assert rtc["readiness_scheduler_guard_steps"] == 1
    assert rtc["readiness_min_guided_overlap_steps"] == 3
    assert runtime["model"]["task_description"] is None
    assert runtime["model"]["require_checkpoint_manifest"] is True
    expected_diagnostics = {
        "rtc_timing": False,
        "rtc_cuda_events": False,
    }
    assert runtime["diagnostics"] == expected_diagnostics
    if path == live_path:
        assert runtime["runtime"] == {"allow_live_joint_state_worker": True}
    else:
        assert "runtime" not in runtime
    assert runtime["joint_names"]["model_joint_order"] == per_arm_model
    assert runtime["joint_names"]["arm_mapping"] == {"l": "left", "r": "right"}
    assert runtime["joint_names"]["state_features"] == ["position"]
    assert runtime["cameras"]["fps"] == 60.0
    assert runtime["cameras"]["mapping"] == expected_cameras
    assert runtime["arms"]["left"]["action_start"] == 0
    assert runtime["arms"]["left"]["action_end"] == 8
    assert runtime["arms"]["right"]["action_start"] == 8
    assert runtime["arms"]["right"]["action_end"] == 16
    for arm, topic in expected_topics.items():
        assert runtime["arms"][arm]["command_topic"] == topic
    expected_max_delta = 0.04 if path == shadow_path else 0.02
    assert runtime["safety"]["max_position_delta"] == expected_max_delta
    assert runtime["safety"]["min_position_delta"] is None
    assert runtime["safety"]["joint_limit_tolerance"] == 0.000001
    reviewed_saturation_margins = {
        "follower_l_finger_joint1": [0.01, 0.01],
        "follower_r_finger_joint1": [0.01, 0.01],
        "follower_l_joint3": [0.03, 0.0],
        "follower_l_joint5": [0.02, 0.0],
        "follower_l_joint6": [0.12, 0.0],
        "follower_l_joint7": [0.025, 0.0],
        "follower_r_joint2": [0.005, 0.0],
        "follower_r_joint6": [0.005, 0.0],
    }
    assert runtime["safety"].get(
        "allow_live_joint_limit_saturation", False
    ) is (path == live_path)
    assert runtime["safety"].get(
        "saturate_all_raw_targets_to_joint_limits", False
    ) is (path == live_path)
    if path == shadow_path:
        assert all(
            arm["command_topic"].startswith("/debug/")
            for arm in runtime["arms"].values()
        )
    expected_saturation_margins = (
        {} if path == live_path else reviewed_saturation_margins
    )
    assert runtime["safety"]["joint_limit_saturation_margins"] == expected_saturation_margins
    assert runtime["safety"]["joint_position_limits"] == expected_joint_limits
    assert runtime["watchdog"] == {
        "camera_timeout_sec": 0.25,
        "joint_state_timeout_sec": 0.10,
        "max_sensor_skew_sec": 0.12,
        "max_action_age_sec": 1.65,
        "startup_grace_sec": 10.0,
    }

print("[preflight] Checkpoint and runtime contracts: PASS")
PY

tokenizer_snapshot="${HF_CACHE}/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
tokenizer_json="${tokenizer_snapshot}/tokenizer.json"
if [[ ! -f "${tokenizer_json}" ]]; then
  echo "ERROR: pinned PaliGemma tokenizer missing: ${tokenizer_json}" >&2
  exit 1
fi
echo "ef6773c135b77b834de1d13c75a4c98ab7a3684ffd602d1831e1f1bf5467c563  ${tokenizer_json}" | sha256sum -c -

python3 - "${DEPLOY_DIR}/cyclonedds_two_pc_gpu.xml.template" \
  "${DEPLOY_DIR}/cyclonedds_two_pc_gpu.xml" \
  "${DDS_IFACE}" "${DDS_LOCAL_IP}" "${DDS_PEER_IP}" <<'PY'
import ipaddress
import re
import sys
from pathlib import Path

template_path, output_path, interface, local_ip, peer_ip = sys.argv[1:]
if not re.fullmatch(r"[A-Za-z0-9_.:-]+", interface):
    raise SystemExit(f"ERROR: invalid DDS interface name: {interface!r}")
ipaddress.ip_address(local_ip)
ipaddress.ip_address(peer_ip)
payload = Path(template_path).read_text()
payload = payload.replace("@DDS_IFACE@", interface)
payload = payload.replace("@DDS_LOCAL_IP@", local_ip)
payload = payload.replace("@DDS_PEER_IP@", peer_ip)
if "@DDS_" in payload:
    raise SystemExit("ERROR: unresolved DDS template placeholder")
Path(output_path).write_text(payload)
PY

if ! grep -q "NetworkInterface name=\"${DDS_IFACE}\"" "${DEPLOY_DIR}/cyclonedds_two_pc_gpu.xml" \
  || ! grep -q "Peer address=\"${DDS_LOCAL_IP}\"" "${DEPLOY_DIR}/cyclonedds_two_pc_gpu.xml" \
  || ! grep -q "Peer address=\"${DDS_PEER_IP}\"" "${DEPLOY_DIR}/cyclonedds_two_pc_gpu.xml" \
  || ! grep -q '<AllowMulticast>false</AllowMulticast>' "${DEPLOY_DIR}/cyclonedds_two_pc_gpu.xml" \
  || ! grep -q '<ParticipantIndex>auto</ParticipantIndex>' "${DEPLOY_DIR}/cyclonedds_two_pc_gpu.xml" \
  || ! grep -q '<MaxAutoParticipantIndex>31</MaxAutoParticipantIndex>' "${DEPLOY_DIR}/cyclonedds_two_pc_gpu.xml"; then
  echo "ERROR: rendered CycloneDDS profile does not match the configured unicast pair" >&2
  exit 1
fi

if ! ip -4 -o addr show dev "${DDS_IFACE}" | grep -Fq "${DDS_LOCAL_IP}/"; then
  echo "ERROR: ${DDS_IFACE} does not currently own ${DDS_LOCAL_IP}" >&2
  exit 1
fi
if ! ping -I "${DDS_IFACE}" -c 2 -W 1 "${DDS_PEER_IP}" >/dev/null; then
  echo "ERROR: DDS peer ${DDS_PEER_IP} is not reachable through ${DDS_IFACE}" >&2
  exit 1
fi

command -v docker >/dev/null || { echo "ERROR: docker is not installed" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is not installed" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

resolved_compose="$(mktemp)"
trap 'rm -f "${resolved_compose}"' EXIT

CONFIG_FILE="${DEPLOY_DIR}/inference_envelope_ckpt000500_shadow.yaml" \
  docker compose \
    --env-file "${ENV_FILE}" \
    -f "${DEPLOY_DIR}/docker-compose.gpu.yml" \
    --profile monitor \
    --profile tools \
    config >"${resolved_compose}"

if ! grep -q 'source: .*cyclonedds_two_pc_gpu.xml' "${resolved_compose}" \
  || ! grep -q 'target: /workspace/configs/cyclonedds/two_pc_gpu.xml' "${resolved_compose}"; then
  echo "ERROR: resolved Compose config does not explicitly mount the reviewed DDS XML" >&2
  exit 1
fi

python3 - "${resolved_compose}" <<'PY'
import os
import sys
import yaml

compose = yaml.safe_load(open(sys.argv[1]))
for service_name in ("inference", "inference-monitor", "dds-check"):
    service = compose["services"][service_name]
    assert service["network_mode"] == "host", service_name
    assert service.get("ipc") != "host", service_name
    environment = compose["services"][service_name]["environment"]
    assert environment["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "SYSTEM_DEFAULT", service_name
    assert environment["CYCLONEDDS_URI"] == (
        "file:///workspace/configs/cyclonedds/two_pc_gpu.xml"
    ), service_name
assert compose["services"]["inference"]["environment"]["CAPTURE_DEBUG_IMAGES"] == "false"
expected_joint_worker = os.environ.get("JOINT_STATE_WORKER", "false").lower()
assert expected_joint_worker in {"false", "true"}
assert (
    compose["services"]["inference"]["environment"]["JOINT_STATE_WORKER"]
    == expected_joint_worker
)
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    assert compose["services"]["inference"]["environment"][variable] == "4", variable
assert int(compose["services"]["inference"]["shm_size"]) == 2 * 1024**3
PY

source_root="$(cd "${DEPLOY_DIR}/../../.." && pwd)"
if [[ ! -f "${source_root}/ros2/src/lerobot_control/lerobot_control/input_watchdog.py" ]]; then
  echo "ERROR: isolated source does not contain the fail-closed input watchdog" >&2
  exit 1
fi
if ! grep -q '_read_consistent_snapshot' \
    "${source_root}/ros2/src/lerobot_control/lerobot_control/shared_image_buffer.py" \
  || ! grep -q 'libatomic1' "${source_root}/docker/inference/Dockerfile"; then
  echo "ERROR: isolated source does not contain the coherent camera snapshot implementation" >&2
  exit 1
fi
if [[ ! -x "${DEPLOY_DIR}/check_ros_publishers.sh" ]]; then
  echo "ERROR: executable DDS authority checker is missing" >&2
  exit 1
fi
home_gate_script="${source_root}/scripts/prepare_robot_home.py"
home_contract="${source_root}/configs/lerobot_control/robot_home_contract.json"
if [[ ! -f "${home_gate_script}" || ! -f "${home_contract}" ]]; then
  echo "ERROR: guarded robot homing source or contract is missing" >&2
  exit 1
fi
if [[ ! -f "${source_root}/ros2/src/anvil_msgs/srv/ResetArms.srv" \
  || ! -f "${source_root}/ros2/src/anvil_msgs/msg/ArmsResetStatus.msg" ]]; then
  echo "ERROR: robot reset ROS2 interfaces are missing from the isolated image source" >&2
  exit 1
fi
python3 - "${home_gate_script}" "${home_contract}" <<'PY'
import importlib.util
from pathlib import Path
import sys

script = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("prepare_robot_home", script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
contract = module.load_contract(Path(sys.argv[2]))
assert len(contract.joints) == 16
expected = {
    "follower_l_joint2": -0.174,
    "follower_l_joint4": 1.5708,
    "follower_l_finger_joint1": 0.045,
    "follower_r_joint2": 0.174,
    "follower_r_joint4": 1.5708,
    "follower_r_finger_joint1": 0.045,
}
targets = {joint.name: joint.target for joint in contract.joints}
assert all(targets[name] == value for name, value in expected.items())
PY

echo "[preflight] Docker, GPU, tokenizer, DDS, watchdog and guarded homing source: PASS"
