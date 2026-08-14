"""Fail-closed input watchdog for robot inference.

The watchdog deliberately uses local monotonic receive times. ROS message stamps
are useful for synchronization diagnostics, but clock jumps or a repeatedly
published old stamp must never keep action publication enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


@dataclass(frozen=True)
class SensorReading:
    """Monotonic freshness and sequence information for one input stream."""

    name: str
    sequence: int
    last_seen_monotonic: float | None


@dataclass(frozen=True)
class InputSnapshot:
    """Atomic-enough view of the sensor inputs used by one safety check."""

    joint_state: SensorReading
    cameras: tuple[SensorReading, ...]
    missing_joints: tuple[str, ...] = ()
    invalid_joints: tuple[str, ...] = ()
    joint_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservationSequence:
    """Sequence identity of one complete observation passed to the policy."""

    joint_state: int
    cameras: tuple[tuple[str, int], ...]

    @classmethod
    def from_snapshot(cls, snapshot: InputSnapshot) -> ObservationSequence:
        return cls(
            joint_state=snapshot.joint_state.sequence,
            cameras=tuple(sorted((camera.name, camera.sequence) for camera in snapshot.cameras)),
        )


class WatchdogState(StrEnum):
    STARTING = "starting"
    ARMED = "armed"
    LATCHED = "latched"


@dataclass(frozen=True)
class WatchdogResult:
    publish_allowed: bool
    state: WatchdogState
    reason: str
    epoch: int
    fault_transition: bool = False
    armed_transition: bool = False


class InputWatchdog:
    """State machine that latches any input-health fault.

    Thread synchronization is intentionally owned by the caller. This lets the
    inference node make a watchdog state transition and an action-queue clear
    one atomic operation under its safety lock.
    """

    def __init__(
        self,
        *,
        camera_timeout_sec: float,
        joint_state_timeout_sec: float,
        max_sensor_skew_sec: float,
        max_action_age_sec: float,
        startup_grace_sec: float,
        started_at_monotonic: float,
    ) -> None:
        if not isfinite(camera_timeout_sec) or camera_timeout_sec <= 0:
            raise ValueError("camera_timeout_sec must be > 0")
        if not isfinite(joint_state_timeout_sec) or joint_state_timeout_sec <= 0:
            raise ValueError("joint_state_timeout_sec must be > 0")
        if not isfinite(max_sensor_skew_sec) or max_sensor_skew_sec <= 0:
            raise ValueError("max_sensor_skew_sec must be > 0")
        if not isfinite(max_action_age_sec) or max_action_age_sec <= 0:
            raise ValueError("max_action_age_sec must be > 0")
        if not isfinite(startup_grace_sec) or startup_grace_sec < 0:
            raise ValueError("startup_grace_sec must be >= 0")

        self.camera_timeout_sec = camera_timeout_sec
        self.joint_state_timeout_sec = joint_state_timeout_sec
        self.max_sensor_skew_sec = max_sensor_skew_sec
        self.max_action_age_sec = max_action_age_sec
        self.startup_grace_sec = startup_grace_sec
        self.started_at_monotonic = started_at_monotonic

        self.state = WatchdogState.STARTING
        self.epoch = 0
        self.reason = "waiting for a complete, fresh observation"
        self.last_observation: ObservationSequence | None = None
        self._fault_snapshot: InputSnapshot | None = None

    @property
    def publish_allowed(self) -> bool:
        return self.state is WatchdogState.ARMED

    def evaluate(self, snapshot: InputSnapshot, now: float) -> WatchdogResult:
        """Evaluate sensor health and latch if an armed input becomes unsafe."""
        if self.state is WatchdogState.LATCHED:
            return self._result(False)

        issues = self._health_issues(snapshot, now)
        if not issues:
            if self.state is WatchdogState.STARTING:
                self.state = WatchdogState.ARMED
                self.reason = "inputs healthy; waiting for a new observation"
                return self._result(True, armed_transition=True)
            return self._result(True)

        reason = "; ".join(issues)
        if self.state is WatchdogState.ARMED:
            return self.trip(reason, snapshot)

        self.reason = reason
        if now - self.started_at_monotonic >= self.startup_grace_sec:
            return self.trip(f"startup grace expired: {reason}", snapshot)
        return self._result(False)

    def accept_observation(
        self,
        sequence: ObservationSequence,
        snapshot: InputSnapshot | None = None,
    ) -> WatchdogResult:
        """Accept exactly one strictly advancing observation sequence."""
        if self.state is not WatchdogState.ARMED:
            return self._result(False)

        if sequence.joint_state <= 0 or any(counter <= 0 for _, counter in sequence.cameras):
            return self.trip(
                "observation contains an uninitialized input sequence",
                snapshot,
            )

        previous = self.last_observation
        if previous is not None and not self._strictly_advances(previous, sequence):
            return self.trip(
                "repeated or non-advancing observation sequence "
                f"(previous={previous}, current={sequence})",
                snapshot,
            )

        self.last_observation = sequence
        self.reason = "inputs healthy"
        return self._result(True)

    def trip(
        self,
        reason: str,
        snapshot: InputSnapshot | None = None,
    ) -> WatchdogResult:
        """Latch a fault and invalidate all work captured under the old epoch."""
        transitioned = self.state is not WatchdogState.LATCHED
        if transitioned:
            self.epoch += 1
            self._fault_snapshot = snapshot
        self.state = WatchdogState.LATCHED
        self.reason = reason
        return self._result(False, fault_transition=transitioned)

    def rearm(self, snapshot: InputSnapshot, now: float) -> tuple[bool, str]:
        """Explicitly rearm after every required stream has recovered and advanced."""
        if self.state is not WatchdogState.LATCHED:
            return False, f"watchdog is {self.state.value}, not latched"

        issues = self._health_issues(snapshot, now)
        if issues:
            return False, "inputs are not healthy: " + "; ".join(issues)

        if self._fault_snapshot is not None:
            fault_sequence = ObservationSequence.from_snapshot(self._fault_snapshot)
            current_sequence = ObservationSequence.from_snapshot(snapshot)
            if not self._strictly_advances(fault_sequence, current_sequence):
                return False, "every input must publish a new sample after the fault before rearm"

        self.epoch += 1
        self.state = WatchdogState.ARMED
        self.reason = "rearmed; waiting for a new complete observation"
        self.last_observation = None
        self._fault_snapshot = None
        return True, self.reason

    def is_epoch_current(self, epoch: int) -> bool:
        """Return whether an inference result may still enter an action queue."""
        return self.publish_allowed and epoch == self.epoch

    def authorize_action(
        self,
        *,
        epoch: int | None,
        source_monotonic: float | None,
        now: float,
        snapshot: InputSnapshot | None = None,
    ) -> WatchdogResult:
        """Authorize one queued action or latch an invalid queue state."""
        if self.state is not WatchdogState.ARMED:
            return self._result(False)
        if epoch != self.epoch:
            return self.trip(
                f"queued action epoch {epoch} does not match safety epoch {self.epoch}",
                snapshot,
            )
        if source_monotonic is None:
            return self.trip("queued action has no source observation", snapshot)
        if not isfinite(source_monotonic) or not isfinite(now):
            return self.trip("queued action has an invalid monotonic source time", snapshot)

        age = now - source_monotonic
        if age < 0:
            return self.trip("queued action has an invalid monotonic source time", snapshot)
        if age > self.max_action_age_sec:
            return self.trip(
                f"action source observation is too old "
                f"({age:.3f}s > {self.max_action_age_sec:.3f}s)",
                snapshot,
            )
        return self._result(True)

    def _health_issues(self, snapshot: InputSnapshot, now: float) -> list[str]:
        issues: list[str] = []

        self._append_freshness_issue(
            issues,
            snapshot.joint_state,
            now,
            self.joint_state_timeout_sec,
        )
        for camera in snapshot.cameras:
            self._append_freshness_issue(issues, camera, now, self.camera_timeout_sec)

        if snapshot.missing_joints:
            issues.append("joint state missing: " + ", ".join(snapshot.missing_joints))
        if snapshot.invalid_joints:
            issues.append("joint state non-finite: " + ", ".join(snapshot.invalid_joints))
        if snapshot.joint_errors:
            issues.append("joint state invalid: " + "; ".join(snapshot.joint_errors))

        readings = (snapshot.joint_state, *snapshot.cameras)
        if readings and all(reading.last_seen_monotonic is not None for reading in readings):
            oldest = min(readings, key=lambda reading: reading.last_seen_monotonic)
            newest = max(readings, key=lambda reading: reading.last_seen_monotonic)
            skew = newest.last_seen_monotonic - oldest.last_seen_monotonic
            if skew > self.max_sensor_skew_sec:
                issues.append(
                    f"sensor receive skew {skew:.3f}s exceeds "
                    f"{self.max_sensor_skew_sec:.3f}s "
                    f"(oldest={oldest.name} seq={oldest.sequence} "
                    f"age={now - oldest.last_seen_monotonic:.3f}s, "
                    f"newest={newest.name} seq={newest.sequence} "
                    f"age={now - newest.last_seen_monotonic:.3f}s)"
                )
        return issues

    @staticmethod
    def _append_freshness_issue(
        issues: list[str],
        reading: SensorReading,
        now: float,
        timeout: float,
    ) -> None:
        if reading.sequence <= 0 or reading.last_seen_monotonic is None:
            issues.append(f"{reading.name} missing")
            return

        if not isfinite(reading.last_seen_monotonic) or not isfinite(now):
            issues.append(f"{reading.name} has an invalid monotonic timestamp")
            return

        age = now - reading.last_seen_monotonic
        if age < 0:
            issues.append(f"{reading.name} has an invalid monotonic timestamp")
        elif age > timeout:
            issues.append(f"{reading.name} stale ({age:.3f}s > {timeout:.3f}s)")

    @staticmethod
    def _strictly_advances(
        previous: ObservationSequence,
        current: ObservationSequence,
    ) -> bool:
        if current.joint_state <= previous.joint_state:
            return False
        if tuple(name for name, _ in current.cameras) != tuple(
            name for name, _ in previous.cameras
        ):
            return False
        return all(
            current_counter > previous_counter
            for (_, previous_counter), (_, current_counter) in zip(
                previous.cameras, current.cameras, strict=True
            )
        )

    def _result(
        self,
        publish_allowed: bool,
        *,
        fault_transition: bool = False,
        armed_transition: bool = False,
    ) -> WatchdogResult:
        return WatchdogResult(
            publish_allowed=publish_allowed,
            state=self.state,
            reason=self.reason,
            epoch=self.epoch,
            fault_transition=fault_transition,
            armed_transition=armed_transition,
        )
