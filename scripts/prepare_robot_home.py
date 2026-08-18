#!/usr/bin/env python3
"""Request the existing arms reset and verify the measured start pose."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JointTarget:
    name: str
    target: float
    position_tolerance: float
    velocity_tolerance: float


@dataclass(frozen=True)
class HomeContract:
    reset_service: str
    reset_status_topic: str
    joint_state_topic: str
    success_status: str
    failure_status_keywords: tuple[str, ...]
    service_wait_timeout_sec: float
    overall_timeout_sec: float
    settle_duration_sec: float
    minimum_settle_samples: int
    joints: tuple[JointTarget, ...]


@dataclass(frozen=True)
class SampleEvaluation:
    ok: bool
    reason: str
    max_position_error: float = math.inf
    max_abs_velocity: float = math.inf


def _positive_float(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and > 0")
    return result


def load_contract(path: Path) -> HomeContract:
    raw = json.loads(path.read_text(encoding="utf-8"))
    joints = tuple(
        JointTarget(
            name=str(item["name"]),
            target=float(item["target"]),
            position_tolerance=_positive_float(
                item["position_tolerance"],
                f"joints[{index}].position_tolerance",
            ),
            velocity_tolerance=_positive_float(
                item["velocity_tolerance"],
                f"joints[{index}].velocity_tolerance",
            ),
        )
        for index, item in enumerate(raw["joints"])
    )
    names = [joint.name for joint in joints]
    if len(joints) != 16:
        raise ValueError(f"home contract must contain exactly 16 joints, got {len(joints)}")
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("home contract joint names must be non-empty and unique")
    if any(not math.isfinite(joint.target) for joint in joints):
        raise ValueError("home contract targets must be finite")

    minimum_samples = int(raw["minimum_settle_samples"])
    if minimum_samples <= 0:
        raise ValueError("minimum_settle_samples must be > 0")
    failure_keywords = tuple(str(value).lower() for value in raw["failure_status_keywords"])
    if not failure_keywords or any(not value for value in failure_keywords):
        raise ValueError("failure_status_keywords must be non-empty")

    return HomeContract(
        reset_service=str(raw["reset_service"]),
        reset_status_topic=str(raw["reset_status_topic"]),
        joint_state_topic=str(raw["joint_state_topic"]),
        success_status=str(raw["success_status"]),
        failure_status_keywords=failure_keywords,
        service_wait_timeout_sec=_positive_float(
            raw["service_wait_timeout_sec"], "service_wait_timeout_sec"
        ),
        overall_timeout_sec=_positive_float(raw["overall_timeout_sec"], "overall_timeout_sec"),
        settle_duration_sec=_positive_float(raw["settle_duration_sec"], "settle_duration_sec"),
        minimum_settle_samples=minimum_samples,
        joints=joints,
    )


def evaluate_joint_sample(
    names: list[str],
    positions: list[float],
    velocities: list[float],
    contract: HomeContract,
) -> SampleEvaluation:
    if len(names) != len(set(names)):
        return SampleEvaluation(False, "joint state contains duplicate names")
    if len(positions) != len(names):
        return SampleEvaluation(False, "joint position array length does not match names")
    if len(velocities) != len(names):
        return SampleEvaluation(False, "joint velocity array length does not match names")

    index_by_name = {name: index for index, name in enumerate(names)}
    missing = [joint.name for joint in contract.joints if joint.name not in index_by_name]
    if missing:
        return SampleEvaluation(False, f"joint state is missing required joints: {missing}")

    max_position_error = 0.0
    max_abs_velocity = 0.0
    for joint in contract.joints:
        index = index_by_name[joint.name]
        position = float(positions[index])
        velocity = float(velocities[index])
        if not math.isfinite(position) or not math.isfinite(velocity):
            return SampleEvaluation(False, f"non-finite state for {joint.name}")
        position_error = abs(position - joint.target)
        abs_velocity = abs(velocity)
        max_position_error = max(max_position_error, position_error)
        max_abs_velocity = max(max_abs_velocity, abs_velocity)
        if position_error > joint.position_tolerance:
            return SampleEvaluation(
                False,
                f"{joint.name} position error {position_error:.6f} exceeds "
                f"{joint.position_tolerance:.6f}",
                max_position_error,
                max_abs_velocity,
            )
        if abs_velocity > joint.velocity_tolerance:
            return SampleEvaluation(
                False,
                f"{joint.name} velocity {abs_velocity:.6f} exceeds {joint.velocity_tolerance:.6f}",
                max_position_error,
                max_abs_velocity,
            )

    return SampleEvaluation(
        True, "pose and velocity are within the home contract", max_position_error, max_abs_velocity
    )


class HomingTransition:
    """Accept success only after a post-request non-success reset state."""

    def __init__(self, contract: HomeContract) -> None:
        self._contract = contract
        self.saw_active = False
        self.complete = False
        self.failure = ""
        self.last_state = ""

    def observe(self, state: str, is_dehome: bool) -> None:
        normalized = state.strip().lower()
        self.last_state = normalized
        if is_dehome:
            self.failure = "reset status unexpectedly entered dehome mode"
            return
        if any(keyword in normalized for keyword in self._contract.failure_status_keywords):
            self.failure = f"reset status reported failure state: {state!r}"
            return
        if normalized == self._contract.success_status.lower():
            if self.saw_active:
                self.complete = True
            return
        if normalized:
            self.saw_active = True


class PoseSettler:
    def __init__(self, contract: HomeContract) -> None:
        self._contract = contract
        self.valid_since: float | None = None
        self.valid_samples = 0
        self.last_evaluation = SampleEvaluation(False, "no post-home joint sample received")

    def observe(
        self,
        received_at: float,
        names: list[str],
        positions: list[float],
        velocities: list[float],
    ) -> None:
        evaluation = evaluate_joint_sample(names, positions, velocities, self._contract)
        self.last_evaluation = evaluation
        if not evaluation.ok:
            self.valid_since = None
            self.valid_samples = 0
            return
        if self.valid_since is None:
            self.valid_since = received_at
        self.valid_samples += 1

    def ready(self, now: float) -> bool:
        return (
            self.valid_since is not None
            and now - self.valid_since >= self._contract.settle_duration_sec
            and self.valid_samples >= self._contract.minimum_settle_samples
        )


def run_ros_gate(contract: HomeContract) -> int:
    import rclpy
    from anvil_msgs.msg import ArmsResetStatus
    from anvil_msgs.srv import ResetArms
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node("inference_home_gate")
    transition = HomingTransition(contract)
    settler = PoseSettler(contract)
    request_started = False
    last_logged_status: tuple[str, bool] | None = None

    status_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    joint_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )

    def on_status(message: ArmsResetStatus) -> None:
        nonlocal last_logged_status
        if not request_started:
            return
        current = (message.state, bool(message.is_dehome))
        if current != last_logged_status:
            print(
                f"HOME_GATE_STATUS state={message.state!r} is_dehome={bool(message.is_dehome)}",
                flush=True,
            )
            last_logged_status = current
        transition.observe(message.state, bool(message.is_dehome))

    def on_joint_state(message: JointState) -> None:
        if not transition.complete:
            return
        settler.observe(
            time.monotonic(),
            list(message.name),
            list(message.position),
            list(message.velocity),
        )

    node.create_subscription(
        ArmsResetStatus,
        contract.reset_status_topic,
        on_status,
        status_qos,
    )
    node.create_subscription(
        JointState,
        contract.joint_state_topic,
        on_joint_state,
        joint_qos,
    )
    client = node.create_client(ResetArms, contract.reset_service)

    try:
        if not client.wait_for_service(timeout_sec=contract.service_wait_timeout_sec):
            raise RuntimeError(
                f"reset service unavailable after {contract.service_wait_timeout_sec:.1f}s: "
                f"{contract.reset_service}"
            )

        # Drain discovery and any already queued volatile samples before the
        # request so a previous 'homed' state cannot satisfy this rollout.
        drain_deadline = time.monotonic() + 0.5
        while time.monotonic() < drain_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)

        request = ResetArms.Request()
        request.dehome = False
        request_started = True
        print(
            f"HOME_GATE_REQUEST service={contract.reset_service} dehome=false",
            flush=True,
        )
        future = client.call_async(request)
        deadline = time.monotonic() + contract.overall_timeout_sec
        response_checked = False

        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if transition.failure:
                raise RuntimeError(transition.failure)
            if future.done() and not response_checked:
                response = future.result()
                if response is None:
                    raise RuntimeError("reset service completed without a response")
                if not response.accepted:
                    raise RuntimeError(f"reset service rejected the request: {response.message}")
                print(f"HOME_GATE_ACCEPTED message={response.message!r}", flush=True)
                response_checked = True
            if response_checked and transition.complete and settler.ready(time.monotonic()):
                evaluation = settler.last_evaluation
                print(
                    "HOME_GATE_PASS "
                    f"samples={settler.valid_samples} "
                    f"max_position_error={evaluation.max_position_error:.6f} "
                    f"max_abs_velocity={evaluation.max_abs_velocity:.6f}",
                    flush=True,
                )
                return 0

        details = (
            f"last_status={transition.last_state!r} "
            f"saw_active={transition.saw_active} "
            f"service_response={response_checked} "
            f"settle_samples={settler.valid_samples} "
            f"settle_reason={settler.last_evaluation.reason}"
        )
        raise RuntimeError(
            f"home gate timed out after {contract.overall_timeout_sec:.1f}s: {details}"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract = load_contract(args.contract)
        return run_ros_gate(contract)
    except Exception as exc:
        print(f"HOME_GATE_FAIL: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
