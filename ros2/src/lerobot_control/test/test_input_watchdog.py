"""Unit tests for the inference input safety state machine."""

import pytest
from lerobot_control.input_watchdog import (
    InputSnapshot,
    InputWatchdog,
    ObservationSequence,
    SensorReading,
    WatchdogState,
)


def snapshot(
    *,
    now: float,
    joint_sequence: int = 1,
    joint_age: float = 0.0,
    camera_sequences: tuple[int, ...] = (1, 1, 1),
    camera_ages: tuple[float, ...] = (0.0, 0.0, 0.0),
    missing_joints: tuple[str, ...] = (),
    invalid_joints: tuple[str, ...] = (),
    joint_errors: tuple[str, ...] = (),
) -> InputSnapshot:
    cameras = tuple(
        SensorReading(
            name=f"camera:{name}",
            sequence=sequence,
            last_seen_monotonic=None if sequence == 0 else now - age,
        )
        for name, sequence, age in zip(
            ("base", "left_wrist", "right_wrist"),
            camera_sequences,
            camera_ages,
            strict=True,
        )
    )
    return InputSnapshot(
        joint_state=SensorReading(
            name="joint_states",
            sequence=joint_sequence,
            last_seen_monotonic=None if joint_sequence == 0 else now - joint_age,
        ),
        cameras=cameras,
        missing_joints=missing_joints,
        invalid_joints=invalid_joints,
        joint_errors=joint_errors,
    )


def watchdog(started_at: float = 0.0) -> InputWatchdog:
    return InputWatchdog(
        camera_timeout_sec=0.25,
        joint_state_timeout_sec=0.10,
        max_sensor_skew_sec=0.10,
        max_action_age_sec=1.50,
        startup_grace_sec=1.0,
        started_at_monotonic=started_at,
    )


def arm(instance: InputWatchdog, now: float = 0.1) -> InputSnapshot:
    healthy = snapshot(now=now)
    result = instance.evaluate(healthy, now)
    assert result.publish_allowed
    assert result.state is WatchdogState.ARMED
    return healthy


def test_startup_without_inputs_is_fail_closed_then_latches() -> None:
    instance = watchdog()
    missing = snapshot(now=0.5, joint_sequence=0, camera_sequences=(0, 0, 0))

    waiting = instance.evaluate(missing, 0.5)
    assert not waiting.publish_allowed
    assert waiting.state is WatchdogState.STARTING

    expired = instance.evaluate(missing, 1.01)
    assert not expired.publish_allowed
    assert expired.state is WatchdogState.LATCHED
    assert expired.fault_transition
    assert expired.epoch == 1


@pytest.mark.parametrize(
    ("joint_age", "camera_ages", "expected"),
    [
        (0.11, (0.0, 0.0, 0.0), "joint_states stale"),
        (0.0, (0.0, 0.26, 0.0), "camera:left_wrist stale"),
    ],
)
def test_stale_input_latches(
    joint_age: float,
    camera_ages: tuple[float, ...],
    expected: str,
) -> None:
    instance = watchdog()
    arm(instance)

    result = instance.evaluate(
        snapshot(now=1.0, joint_age=joint_age, camera_ages=camera_ages),
        1.0,
    )

    assert result.state is WatchdogState.LATCHED
    assert expected in result.reason
    assert not result.publish_allowed


def test_missing_or_non_finite_required_joint_latches() -> None:
    for kwargs, expected in (
        ({"missing_joints": ("follower_l_joint2",)}, "joint state missing"),
        ({"invalid_joints": ("follower_r_joint4",)}, "joint state non-finite"),
        ({"joint_errors": ("duplicate joint names",)}, "joint state invalid"),
    ):
        instance = watchdog()
        arm(instance)
        result = instance.evaluate(snapshot(now=0.2, **kwargs), 0.2)
        assert result.state is WatchdogState.LATCHED
        assert expected in result.reason


def test_sensor_receive_skew_latches_even_when_each_input_is_fresh() -> None:
    instance = watchdog()
    arm(instance)

    result = instance.evaluate(
        snapshot(now=0.2, joint_age=0.0, camera_ages=(0.0, 0.0, 0.101)),
        0.2,
    )

    assert result.state is WatchdogState.LATCHED
    assert "sensor receive skew" in result.reason
    assert "oldest=camera:right_wrist seq=1 age=0.101s" in result.reason
    assert "newest=joint_states seq=1 age=0.000s" in result.reason


def test_repeated_observation_sequence_latches() -> None:
    instance = watchdog()
    healthy = arm(instance)
    sequence = ObservationSequence.from_snapshot(healthy)
    assert instance.accept_observation(sequence, healthy).publish_allowed

    repeated = instance.accept_observation(sequence, healthy)

    assert repeated.state is WatchdogState.LATCHED
    assert "non-advancing" in repeated.reason


def test_latched_fault_never_auto_resumes_and_requires_all_inputs_to_advance() -> None:
    instance = watchdog()
    healthy = arm(instance)
    instance.trip("test fault", healthy)

    auto = instance.evaluate(snapshot(now=0.2, joint_sequence=2, camera_sequences=(2, 2, 2)), 0.2)
    assert auto.state is WatchdogState.LATCHED
    assert not auto.publish_allowed

    unchanged_camera = snapshot(
        now=0.2,
        joint_sequence=2,
        camera_sequences=(2, 1, 2),
    )
    success, message = instance.rearm(unchanged_camera, 0.2)
    assert not success
    assert "every input" in message

    recovered = snapshot(now=0.3, joint_sequence=2, camera_sequences=(2, 2, 2))
    success, _ = instance.rearm(recovered, 0.3)
    assert success
    assert instance.state is WatchdogState.ARMED
    assert instance.epoch == 2


def test_epoch_discards_in_flight_result_across_fault_and_rearm() -> None:
    instance = watchdog()
    healthy = arm(instance)
    in_flight_epoch = instance.epoch
    assert instance.is_epoch_current(in_flight_epoch)

    instance.trip("camera stopped", healthy)
    assert not instance.is_epoch_current(in_flight_epoch)

    recovered = snapshot(now=0.3, joint_sequence=2, camera_sequences=(2, 2, 2))
    success, _ = instance.rearm(recovered, 0.3)
    assert success
    assert not instance.is_epoch_current(in_flight_epoch)
    assert instance.is_epoch_current(instance.epoch)


def test_action_older_than_limit_latches_before_publication() -> None:
    instance = watchdog()
    healthy = arm(instance)

    result = instance.authorize_action(
        epoch=instance.epoch,
        source_monotonic=10.0,
        now=11.501,
        snapshot=healthy,
    )

    assert result.state is WatchdogState.LATCHED
    assert "too old" in result.reason
    assert not result.publish_allowed


@pytest.mark.parametrize("source_monotonic", [float("nan"), float("inf")])
def test_non_finite_action_source_time_latches(source_monotonic: float) -> None:
    instance = watchdog()
    healthy = arm(instance)

    result = instance.authorize_action(
        epoch=instance.epoch,
        source_monotonic=source_monotonic,
        now=1.0,
        snapshot=healthy,
    )

    assert result.state is WatchdogState.LATCHED
    assert "invalid monotonic" in result.reason


def test_non_finite_sensor_receipt_time_never_arms() -> None:
    instance = watchdog()
    unhealthy = InputSnapshot(
        joint_state=SensorReading("joint_states", 1, float("nan")),
        cameras=(SensorReading("camera:base", 1, 0.1),),
    )

    result = instance.evaluate(unhealthy, 0.1)

    assert not result.publish_allowed
    assert "invalid monotonic" in result.reason


def test_invalid_configuration_rejected() -> None:
    with pytest.raises(ValueError):
        InputWatchdog(
            camera_timeout_sec=0,
            joint_state_timeout_sec=0.1,
            max_sensor_skew_sec=0.1,
            max_action_age_sec=1.5,
            startup_grace_sec=1.0,
            started_at_monotonic=0.0,
        )
