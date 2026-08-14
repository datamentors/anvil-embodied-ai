"""Exact sensor provenance tests for the multi-process observation strategy."""

from types import SimpleNamespace

import numpy as np
import pytest
from lerobot_control.strategies import multi_process
from lerobot_control.strategies.multi_process import MultiProcessStrategy

CAMERA_NAMES = ["base", "left_wrist", "right_wrist"]


class _AdvancingImageBuffer:
    """Return captured frames while advancing live metadata between reads."""

    def __init__(
        self,
        strategy: MultiProcessStrategy,
        camera_received: tuple[float, float, float],
        *,
        next_joint_received: float = 999.0,
    ) -> None:
        self._strategy = strategy
        self._camera_received = camera_received
        self._next_joint_received = next_joint_received
        self.metadata_reads = 0

    def read_all_if_ready_with_metadata(self):
        # Deterministic race interleaving: get_observation has already copied
        # joint sample 1 under the lock. Simulate callback sample 2 arriving
        # while the camera snapshot is being collected.
        with self._strategy._joint_lock:
            self._strategy._joint_positions = {"follower_l_joint1": 9.0}
            self._strategy._joint_received_monotonic = self._next_joint_received
            self._strategy._joint_sequence = 2
            self._strategy._joint_errors = ()

        return {
            camera_name: (
                np.full((2, 2, 3), index, dtype=np.uint8),
                100.0 + index,
                index,
                received,
            )
            for index, (camera_name, received) in enumerate(
                zip(CAMERA_NAMES, self._camera_received, strict=True),
                start=1,
            )
        }

    def get_frame_metadata(self):
        self.metadata_reads += 1
        return dict.fromkeys(CAMERA_NAMES, (100, 999.0))

    def has_new_frame(self, _camera_name: str) -> bool:
        return True


def _strategy_with_joint_sample(
    *,
    joint_received: float,
    camera_received: tuple[float, float, float],
    next_joint_received: float = 999.0,
    max_sensor_skew: float = 0.10,
) -> tuple[MultiProcessStrategy, _AdvancingImageBuffer]:
    strategy = MultiProcessStrategy()
    strategy._joint_names_config = {
        "observation_prefix": "follower",
        "separator": "_",
        "arm_mapping": {"l": "left"},
        "model_joint_order": ["joint1"],
        "state_features": ["position"],
    }
    strategy._joint_positions = {"follower_l_joint1": 1.25}
    strategy._joint_received_monotonic = joint_received
    strategy._joint_sequence = 1
    strategy._max_sensor_skew_sec = max_sensor_skew
    image_buffer = _AdvancingImageBuffer(
        strategy,
        camera_received,
        next_joint_received=next_joint_received,
    )
    strategy._image_buffer = image_buffer
    return strategy, image_buffer


@pytest.mark.parametrize(
    ("joint_received", "camera_received", "expected_source"),
    [
        (10.0, (10.02, 10.03, 10.04), 10.0),
        (10.05, (10.02, 10.03, 10.04), 10.02),
        (10.05, (10.02, 10.005, 10.04), 10.005),
    ],
)
def test_observation_source_is_oldest_exact_consumed_input_during_metadata_race(
    joint_received,
    camera_received,
    expected_source,
) -> None:
    strategy, image_buffer = _strategy_with_joint_sample(
        joint_received=joint_received,
        camera_received=camera_received,
    )

    observation = strategy.get_observation(CAMERA_NAMES)

    assert observation is not None
    assert observation["observation.state"].item() == 1.25
    assert strategy.get_last_observation_monotonic() == expected_source
    assert strategy.get_last_observation_sequence().joint_state == 1
    assert strategy.get_last_observation_sequence().cameras == (
        ("base", 1),
        ("left_wrist", 2),
        ("right_wrist", 3),
    )
    assert image_buffer.metadata_reads == 0

    # Live joint metadata advanced during the camera read, but neither the
    # state nor provenance of observation 1 was replaced by sample 2.
    assert strategy._joint_sequence == 2
    assert strategy._joint_received_monotonic == 999.0


def test_exact_invalid_joint_sample_is_not_hidden_by_new_valid_sample() -> None:
    strategy, _image_buffer = _strategy_with_joint_sample(
        joint_received=10.0,
        camera_received=(10.02, 10.03, 10.04),
        next_joint_received=10.04,
    )
    strategy._joint_positions = {}
    strategy._joint_errors = ("name/position length mismatch (2 != 1)",)

    with pytest.raises(
        ValueError,
        match=r"joint sample 1 is invalid: name/position length mismatch \(2 != 1\)",
    ):
        strategy.get_observation(CAMERA_NAMES)

    # The callback raced after sample 1 was copied and made the live snapshot
    # valid. Validation must still describe and reject the consumed sample 1.
    assert strategy._joint_sequence == 2
    assert strategy._joint_errors == ()
    assert strategy._joint_received_monotonic == 10.04
    assert strategy.get_last_observation_sequence() is None
    assert strategy.get_last_observation_monotonic() is None


def test_exact_sensor_skew_is_not_hidden_by_new_synchronized_joint_sample() -> None:
    strategy, _image_buffer = _strategy_with_joint_sample(
        joint_received=10.0,
        camera_received=(10.20, 10.21, 10.22),
        # The live joint sample advances to the camera window during the read,
        # but sample 1 is still the state paired with these exact frames.
        next_joint_received=10.21,
        max_sensor_skew=0.10,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"exact sensor receive skew 0\.220s exceeds 0\.100s "
            r"\(oldest=joint_states, newest=camera:right_wrist\)"
        ),
    ):
        strategy.get_observation(CAMERA_NAMES)

    assert strategy._joint_sequence == 2
    assert strategy._joint_received_monotonic == 10.21
    assert strategy.get_last_observation_sequence() is None


def test_setup_reads_exact_skew_limit_from_watchdog_config(monkeypatch) -> None:
    strategy = MultiProcessStrategy()
    monkeypatch.setattr(strategy, "_setup_shared_memory", lambda: None)
    monkeypatch.setattr(strategy, "_start_workers", lambda: None)
    monkeypatch.setattr(strategy, "_setup_joint_subscription", lambda _topic: None)
    logger = SimpleNamespace(info=lambda _message: None)

    strategy.setup(
        node=SimpleNamespace(get_logger=lambda: logger),
        config={"watchdog": {"max_sensor_skew_sec": 0.075}},
        camera_mapping={},
        joint_names_config={},
        joint_state_topic="/joint_states",
        image_shape=(2, 2, 3),
    )

    assert strategy._max_sensor_skew_sec == 0.075


def test_joint_callback_captures_ingress_before_metrics_work(monkeypatch) -> None:
    strategy = MultiProcessStrategy()
    monotonic_values = iter((10.0, 20.0))
    monkeypatch.setattr(
        multi_process.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    class _Metrics:
        metric_time = None

        def record_joint_state(self) -> None:
            self.metric_time = multi_process.time.monotonic()

    metrics = _Metrics()
    strategy._metrics = metrics
    message = SimpleNamespace(
        name=["follower_l_joint1"],
        position=[1.25],
        velocity=[],
        effort=[],
        header=SimpleNamespace(stamp=SimpleNamespace(sec=123, nanosec=456)),
    )

    strategy._joint_callback(message)

    assert metrics.metric_time == 20.0
    with strategy._joint_lock:
        assert strategy._joint_positions == {"follower_l_joint1": 1.25}
        assert strategy._joint_received_monotonic == 10.0
        assert strategy._joint_sequence == 1


def test_joint_worker_mode_owns_the_only_subscription(monkeypatch) -> None:
    strategy = MultiProcessStrategy()
    setup_subscription = []
    monkeypatch.setattr(strategy, "_setup_shared_memory", lambda: None)
    monkeypatch.setattr(strategy, "_start_workers", lambda: None)
    monkeypatch.setattr(
        strategy,
        "_setup_joint_subscription",
        lambda topic: setup_subscription.append(topic),
    )
    logger = SimpleNamespace(info=lambda _message: None)

    strategy.setup(
        node=SimpleNamespace(get_logger=lambda: logger),
        config={
            "runtime": {"joint_state_worker": True},
            "watchdog": {"max_sensor_skew_sec": 0.1},
            "arms": {"left": {"command_topic": "/debug/left/commands"}},
        },
        camera_mapping={},
        joint_names_config={},
        joint_state_topic="/joint_states",
        image_shape=(2, 2, 3),
    )

    assert strategy._joint_state_worker_enabled is True
    assert setup_subscription == []


def test_joint_worker_mode_requires_explicit_live_opt_in() -> None:
    with pytest.raises(ValueError, match="allow_live_joint_state_worker=true"):
        MultiProcessStrategy._validate_joint_state_worker_mode(
            {"arms": {"left": {"command_topic": "/controller/commands"}}}
        )

    MultiProcessStrategy._validate_joint_state_worker_mode(
        {
            "runtime": {"allow_live_joint_state_worker": True},
            "arms": {"left": {"command_topic": "/controller/commands"}},
        }
    )


def test_joint_worker_live_opt_in_must_be_boolean() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        MultiProcessStrategy._validate_joint_state_worker_mode(
            {
                "runtime": {"allow_live_joint_state_worker": "yes"},
                "arms": {"left": {"command_topic": "/debug/commands"}},
            }
        )


def test_worker_refresh_deserializes_each_counter_once_and_preserves_metrics(
    monkeypatch,
) -> None:
    strategy = MultiProcessStrategy()
    strategy._joint_state_worker_enabled = True
    strategy._joint_worker_process = SimpleNamespace(exitcode=None)
    strategy._joint_buffer = SimpleNamespace(
        read=lambda: (b"serialized-joint-state", 10.25, 500)
    )
    message = SimpleNamespace(
        name=["follower_l_joint1"],
        position=[1.25],
        velocity=[0.5],
        effort=[0.1],
        header=SimpleNamespace(stamp=SimpleNamespace(sec=123, nanosec=456)),
    )
    deserialize_calls = []

    def deserialize(payload, message_type):
        deserialize_calls.append((payload, message_type))
        return message

    class _Metrics:
        deltas = []

        def record_joint_states(self, count):
            self.deltas.append(count)

    metrics = _Metrics()
    strategy._metrics = metrics
    monkeypatch.setattr(multi_process, "deserialize_message", deserialize)

    strategy._refresh_joint_state_from_worker()
    strategy._refresh_joint_state_from_worker()

    assert deserialize_calls == [(b"serialized-joint-state", multi_process.JointState)]
    assert metrics.deltas == [500]
    with strategy._joint_lock:
        assert strategy._joint_positions == {"follower_l_joint1": 1.25}
        assert strategy._joint_velocities == {"follower_l_joint1": 0.5}
        assert strategy._joint_efforts == {"follower_l_joint1": 0.1}
        assert strategy._joint_received_monotonic == 10.25
        assert strategy._joint_sequence == 500


def test_worker_death_fails_before_reusing_cached_joint_state() -> None:
    strategy = MultiProcessStrategy()
    strategy._joint_state_worker_enabled = True
    strategy._joint_worker_process = SimpleNamespace(exitcode=17)
    strategy._joint_buffer = SimpleNamespace(
        read=lambda: pytest.fail("dead worker slot must not be read")
    )
    strategy._joint_positions = {"follower_l_joint1": 1.25}

    with pytest.raises(RuntimeError, match="exited unexpectedly.*exitcode=17"):
        strategy._refresh_joint_state_from_worker()


def test_worker_counter_regression_fails_closed(monkeypatch) -> None:
    strategy = MultiProcessStrategy()
    strategy._joint_state_worker_enabled = True
    strategy._joint_worker_process = SimpleNamespace(exitcode=None)
    strategy._joint_worker_last_counter = 10
    strategy._joint_buffer = SimpleNamespace(read=lambda: (b"old", 10.0, 9))
    monkeypatch.setattr(
        multi_process,
        "deserialize_message",
        lambda *_args: pytest.fail("regressed payload must not be deserialized"),
    )

    with pytest.raises(RuntimeError, match=r"counter regressed \(9 < 10\)"):
        strategy._refresh_joint_state_from_worker()
