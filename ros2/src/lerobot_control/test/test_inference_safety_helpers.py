"""Tests for queue invalidation and action validation without a ROS graph."""

import sys
import threading
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

# The production node runs inside the ROS2 image. These tests exercise methods
# that do not need a ROS graph, so provide import-only types on developer hosts
# where ROS2 is intentionally absent.
try:
    import rclpy  # noqa: F401
except ModuleNotFoundError:
    rclpy_module = ModuleType("rclpy")
    callback_groups = ModuleType("rclpy.callback_groups")
    callback_groups.MutuallyExclusiveCallbackGroup = type(
        "MutuallyExclusiveCallbackGroup", (), {}
    )
    callback_groups.ReentrantCallbackGroup = type("ReentrantCallbackGroup", (), {})
    executors = ModuleType("rclpy.executors")
    executors.MultiThreadedExecutor = type("MultiThreadedExecutor", (), {})
    node_module = ModuleType("rclpy.node")
    node_module.Node = type("Node", (), {})

    std_msgs = ModuleType("std_msgs")
    std_msgs_msg = ModuleType("std_msgs.msg")
    class FakeFloat64MultiArray:
        def __init__(self):
            self.data = []

    std_msgs_msg.Float64MultiArray = FakeFloat64MultiArray
    std_srvs = ModuleType("std_srvs")
    std_srvs_srv = ModuleType("std_srvs.srv")
    std_srvs_srv.Trigger = type("Trigger", (), {})

    sys.modules.update(
        {
            "rclpy": rclpy_module,
            "rclpy.callback_groups": callback_groups,
            "rclpy.executors": executors,
            "rclpy.node": node_module,
            "std_msgs": std_msgs,
            "std_msgs.msg": std_msgs_msg,
            "std_srvs": std_srvs,
            "std_srvs.srv": std_srvs_srv,
        }
    )

from lerobot_control.action_limiter import ActionLimiter
from lerobot_control.inference_node import (
    LeRobotInferenceNode,
    PendingRTCCudaTiming,
    RTCDispatchSnapshot,
    RTCMergeStageTiming,
    VLAObservationTiming,
)
from lerobot_control.input_watchdog import (
    ObservationProvenance,
    ObservationSequence,
    SensorReading,
    WatchdogResult,
    WatchdogState,
)


class FakeQueue:
    def __init__(self) -> None:
        self.items = [1, 2, 3]

    def clear(self) -> None:
        self.items.clear()


class FakeLimiter:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class FakeMetrics:
    def __init__(self) -> None:
        self.control_calls = 0
        self.inference_calls = 0

    def record_control_loop(self) -> None:
        self.control_calls += 1

    def record_inference(self) -> None:
        self.inference_calls += 1

    def record_action_output(self) -> None:
        pass


class PassThroughLimiter:
    def reorder(self, action):
        return action.copy()

    def process(self, action, _current):
        return action.copy()

    def process_controller_order(self, action, _current):
        return action.copy()


class FakePublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(list(message.data))


class FakeLogger:
    def __init__(self) -> None:
        self.info_messages = []
        self.warn_messages = []
        self.error_messages = []

    def info(self, message) -> None:
        self.info_messages.append(message)

    def error(self, _message) -> None:
        self.error_messages.append(_message)

    def warn(self, _message) -> None:
        self.warn_messages.append(_message)

    def warning(self, message) -> None:
        self.warn_messages.append(message)


class FakeCudaEvent:
    def __init__(self, *, ready: bool, elapsed_ms: float = 0.0) -> None:
        self.ready = ready
        self.elapsed_ms = elapsed_ms
        self.query_calls = 0
        self.elapsed_calls = 0

    def query(self) -> bool:
        self.query_calls += 1
        return self.ready

    def elapsed_time(self, _end_event) -> float:
        self.elapsed_calls += 1
        return self.elapsed_ms


class FakeArmedWatchdog:
    def __init__(self) -> None:
        self.publish_allowed = True
        self.trip_reason = None
        self.latched = False
        self.epoch = 0
        self.max_action_age_sec = 1.5

    def is_epoch_current(self, epoch) -> bool:
        return self.publish_allowed and epoch == self.epoch

    def authorize_action(self, **_kwargs) -> WatchdogResult:
        return WatchdogResult(True, WatchdogState.ARMED, "healthy", 0)

    def trip(self, reason, snapshot=None) -> WatchdogResult:
        del snapshot
        transitioned = not self.latched
        self.latched = True
        self.publish_allowed = False
        self.trip_reason = reason
        if transitioned:
            self.epoch += 1
        return WatchdogResult(
            False,
            WatchdogState.LATCHED,
            reason,
            self.epoch,
            fault_transition=transitioned,
        )

    def rearm(self, _snapshot, _now) -> tuple[bool, str]:
        if not self.latched:
            return False, "watchdog is not latched"
        self.epoch += 1
        self.latched = False
        self.publish_allowed = True
        self.trip_reason = None
        return True, "rearmed"


class FakeLatencyTracker:
    def __init__(self) -> None:
        self.values = []
        self.reset_calls = 0

    def add(self, value) -> None:
        self.values.append(value)

    def reset(self) -> None:
        self.values.clear()
        self.reset_calls += 1


class FakeVlaQueue:
    def __init__(self, actions=None, *, populate_on_merge: bool = True) -> None:
        self.actions = deque(actions or [])
        self.populate_on_merge = populate_on_merge
        self.merge_calls = []
        self.get_calls = 0
        self.action_index = 0

    def clear(self) -> None:
        self.actions.clear()
        self.action_index = 0

    def get(self):
        self.get_calls += 1
        if not self.actions:
            return None
        self.action_index += 1
        return self.actions.popleft()

    def qsize(self) -> int:
        return len(self.actions)

    def get_action_index(self) -> int:
        return self.action_index

    def get_left_over(self):
        if not self.actions:
            return None
        return np.stack(tuple(self.actions))

    def merge(self, original, processed, new_delay, action_index_before_inference) -> None:
        self.merge_calls.append(
            (original, processed, new_delay, action_index_before_inference)
        )
        if self.populate_on_merge:
            remaining = max(0, min(len(original), len(processed)) - new_delay)
            self.actions = deque(np.array([0.0]) for _ in range(remaining))
        else:
            self.actions.clear()
        self.action_index = 0


def make_vla_result_node() -> LeRobotInferenceNode:
    """Build the minimum state needed to commit or reject an RTC result."""
    node = object.__new__(LeRobotInferenceNode)
    node.model_type = "pi05"
    node._watchdog = FakeArmedWatchdog()
    node._action_queue = FakeVlaQueue()
    node._action_queue_lock = threading.Lock()
    node._safety_lock = threading.RLock()
    node._model_lock = threading.Lock()
    node._policy_epoch = 0
    node._latency_tracker = FakeLatencyTracker()
    node.metrics = FakeMetrics()
    node._vla_action_source_monotonic = None
    node._vla_action_epoch = None
    node._vla_warmup_pending = True
    node._vla_policy_ready = False
    node._vla_stale_result_count = 0
    node._vla_seeded = False
    node._rtc_threshold = 50
    node._rtc_readiness_guided_forwards = 5
    node._rtc_readiness_guard_steps = 2
    node._rtc_index_phase_tolerance_steps = 1
    node._rtc_scheduler_guard_steps = 1
    node._rtc_min_guided_overlap_steps = 3
    node._rtc_guided_streak = 0
    node._rtc_guided_latencies = deque(maxlen=5)
    node.control_freq = 30.0
    node._classic_action_deque = deque()
    node._classic_chunk_source_monotonic = None
    node._delta_ref_state = None
    node._abs_shadow_queue = deque()
    node.model = SimpleNamespace(
        config=SimpleNamespace(
            rtc_config=SimpleNamespace(execution_horizon=24)
        )
    )
    node._new_vla_action_queue = FakeVlaQueue
    logger = FakeLogger()
    node.get_logger = lambda: logger
    return node


def commit_vla_result(
    node: LeRobotInferenceNode,
    *,
    source: float,
    completed: float,
    elapsed: float = 0.5,
    policy_epoch: int | None = None,
    epoch: int | None = None,
    guided: bool | None = None,
    chunk_size: int = 50,
    dispatch: RTCDispatchSnapshot | None = None,
    runtime: float | None = None,
) -> bool:
    if policy_epoch is None:
        policy_epoch = node._policy_epoch
    if epoch is None:
        epoch = node._watchdog.epoch
    if guided is None:
        guided = node._vla_seeded
    if runtime is None:
        runtime = elapsed
    if dispatch is None:
        dispatch = RTCDispatchSnapshot(
            queue=node._action_queue,
            queue_size=node._action_queue.qsize(),
            action_index=node._action_queue.get_action_index(),
            requested_at_monotonic=completed - runtime,
        )
    original = np.zeros((chunk_size, 1), dtype=np.float32)
    processed = np.zeros((chunk_size, 1), dtype=np.float32)
    with node._safety_lock, patch(
        "lerobot_control.inference_node.time.monotonic", return_value=completed
    ):
        return node._commit_vla_result_locked(
            original=original,
            processed=processed,
            dispatch=dispatch,
            observation_monotonic=source,
            epoch=epoch,
            policy_epoch=policy_epoch,
            elapsed=elapsed,
            completed_at_monotonic=completed,
            guided=guided,
        )


def make_timer_safety_node() -> LeRobotInferenceNode:
    """Build the minimum node state needed to exercise timer fault paths."""
    node = object.__new__(LeRobotInferenceNode)
    node._shutting_down = False
    node.camera_names = ["base"]
    node._safety_lock = threading.RLock()
    node._model_lock = threading.Lock()
    node._policy_epoch = 0
    node._watchdog = FakeArmedWatchdog()
    node._classic_action_deque = deque(
        [(np.array([0.1, 0.2]), 10.0, 0)]
    )
    node._classic_chunk_source_monotonic = 10.0
    node._vla_action_source_monotonic = None
    node._vla_action_epoch = None
    node._delta_ref_state = np.array([0.0, 0.0])
    node._abs_shadow_queue = deque([np.array([0.3, 0.4])])
    node.action_limiter = FakeLimiter()
    node.metrics = FakeMetrics()
    node.get_logger = lambda: FakeLogger()
    node.model_type = "act"
    return node


def find_deploy_config(config_name: str) -> Path:
    """Find source-tree configs both locally and in the runtime image."""
    relative_path = Path("configs") / "lerobot_control" / config_name
    roots = (Path.cwd(), Path("/workspace"), *Path(__file__).resolve().parents)
    for root in roots:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    raise AssertionError(f"could not locate {relative_path}")


def make_two_arm_node() -> LeRobotInferenceNode:
    node = object.__new__(LeRobotInferenceNode)
    node.arms_config = {
        "left": {"action_start": 0, "action_end": 2, "ros_prefix": "follower_l"},
        "right": {"action_start": 2, "action_end": 4, "ros_prefix": "follower_r"},
    }
    node.joint_names_config = {"controller_joint_order": ["joint1", "joint2"]}
    positions = {
        "follower_l_joint1": 0.0,
        "follower_l_joint2": 0.0,
        "follower_r_joint1": 0.0,
        "follower_r_joint2": 0.0,
    }
    node.strategy = SimpleNamespace(get_current_joint_positions=lambda: positions)
    node._joint_position_limits = dict.fromkeys(positions, (-1.0, 1.0))
    node._joint_limit_tolerance = 1e-6
    node._enforce_joint_position_limits = True
    node._saturate_joint_targets = frozenset()
    node._saturate_joint_margins = {}
    node._saturation_counts = {}
    node.action_limiter = PassThroughLimiter()
    node.arm_publishers = {"left": FakePublisher(), "right": FakePublisher()}
    node._monitor_enable = False
    node._smooth_tracker = None
    node._debug = False
    node.metrics = FakeMetrics()
    node._has_published = False
    logger = FakeLogger()
    node.get_logger = lambda: logger
    return node


def test_first_vla_forward_is_always_discarded_as_warmup() -> None:
    node = make_vla_result_node()

    committed = commit_vla_result(
        node,
        source=10.0,
        completed=11.6,
        elapsed=1.6,
    )

    assert committed is False
    assert node._action_queue.merge_calls == []
    assert node._vla_warmup_pending is False
    assert node._vla_policy_ready is False
    assert node._vla_action_source_monotonic is None
    assert node._vla_action_epoch is None
    assert node._watchdog.publish_allowed is True
    assert node._watchdog.latched is False
    assert node._latency_tracker.reset_calls == 1
    assert node._latency_tracker.values == []
    assert node.metrics.inference_calls == 0
    assert any("WARMUP_DISCARDED" in message for message in node.get_logger().info_messages)


def test_post_warmup_result_only_seeds_a_provisional_unpublished_queue() -> None:
    node = make_vla_result_node()
    assert not commit_vla_result(node, source=10.0, completed=11.6, elapsed=1.6)

    committed = commit_vla_result(
        node,
        source=20.0,
        completed=20.5,
        elapsed=0.5,
    )

    assert committed is False
    assert len(node._action_queue.merge_calls) == 1
    assert node._action_queue.merge_calls[0][2:] == (15, None)
    assert node._action_queue.qsize() == 35
    assert node._vla_seeded is True
    assert node._vla_policy_ready is False
    assert node._rtc_guided_streak == 0
    assert node._vla_action_source_monotonic == 20.0
    assert node._vla_action_epoch == 0
    assert node._latency_tracker.values == [0.5]
    assert node.metrics.inference_calls == 1
    assert any("SEED_PROVISIONAL" in message for message in node.get_logger().info_messages)
    assert not any("POLICY_READY" in message for message in node.get_logger().info_messages)


def test_seed_delay_and_tracker_include_dispatch_and_commit_wait() -> None:
    node = make_vla_result_node()
    assert not commit_vla_result(node, source=10.0, completed=11.6, elapsed=1.6)

    assert not commit_vla_result(
        node,
        source=20.0,
        completed=20.8,
        elapsed=0.5,
        runtime=0.61,
    )

    # The model/postprocess span is only 0.5 s and the observation is 0.8 s old.
    # RTC alignment uses neither: it includes dispatch and commit-lock wait in
    # the exact 0.61 s request→merge runtime.
    assert node._action_queue.merge_calls[-1][2:] == (19, None)
    assert node._action_queue.qsize() == 31
    assert node._latency_tracker.values == [pytest.approx(0.61)]


def test_five_consecutive_sustainable_guided_refills_mark_policy_ready() -> None:
    node = make_vla_result_node()
    assert not commit_vla_result(node, source=10.0, completed=12.6, elapsed=2.6)
    assert not commit_vla_result(node, source=20.0, completed=20.55, elapsed=0.55)

    for index in range(4):
        source = 30.0 + index
        assert not commit_vla_result(
            node,
            source=source,
            completed=source + 0.55,
            elapsed=0.55,
        )
        assert node._rtc_guided_streak == index + 1
        assert node._vla_policy_ready is False

    assert commit_vla_result(
        node,
        source=40.0,
        completed=40.55,
        elapsed=0.55,
    )
    assert node._rtc_guided_streak == 5
    assert node._vla_policy_ready is True
    assert node._action_queue.qsize() == 33
    assert node.metrics.inference_calls == 6
    assert any("POLICY_READY" in message for message in node.get_logger().info_messages)


def test_empty_rtc_merge_is_dropped_without_marking_policy_ready() -> None:
    node = make_vla_result_node()
    assert not commit_vla_result(node, source=10.0, completed=11.6, elapsed=1.6)
    node._action_queue.populate_on_merge = False

    committed = commit_vla_result(
        node,
        source=20.0,
        completed=20.5,
        elapsed=0.5,
    )

    assert committed is False
    assert node._watchdog.latched is False
    assert node._watchdog.publish_allowed is True
    assert node._vla_policy_ready is False
    assert node._vla_action_source_monotonic is None
    assert node._vla_action_epoch is None
    assert node._vla_seeded is False
    assert node._action_queue.qsize() == 0
    assert node._vla_stale_result_count == 0
    assert node.metrics.inference_calls == 0
    assert any(
        "EMPTY_RESULT_DISCARDED" in message
        for message in node.get_logger().warn_messages
    )


def test_stale_post_warmup_result_is_dropped_pre_ready_then_newer_result_retries() -> None:
    node = make_vla_result_node()
    assert not commit_vla_result(node, source=10.0, completed=11.6, elapsed=1.6)

    stale_committed = commit_vla_result(
        node,
        source=20.0,
        completed=21.501,
        elapsed=1.501,
    )

    assert stale_committed is False
    assert node._watchdog.publish_allowed is True
    assert node._watchdog.latched is False
    assert node._vla_policy_ready is False
    assert node._vla_stale_result_count == 1
    assert node._action_queue.merge_calls == []
    assert node._latency_tracker.values == []
    assert node.metrics.inference_calls == 0
    assert any(
        "STALE_RESULT_DISCARDED" in message
        for message in node.get_logger().warn_messages
    )

    assert not commit_vla_result(node, source=22.0, completed=22.5, elapsed=0.5)
    assert node._vla_seeded is True
    assert node._vla_policy_ready is False
    assert len(node._action_queue.merge_calls) == 1


def test_stale_pre_ready_refill_drops_the_old_seed_and_its_latency() -> None:
    node = make_vla_result_node()
    assert not commit_vla_result(node, source=10.0, completed=11.6, elapsed=1.6)
    assert not commit_vla_result(node, source=20.0, completed=20.5, elapsed=0.5)
    old_seed_queue = node._action_queue
    assert node._vla_seeded is True
    assert node._latency_tracker.values == [0.5]

    assert not commit_vla_result(
        node,
        source=30.0,
        completed=31.501,
        elapsed=1.0,
        guided=True,
    )

    assert node._action_queue is not old_seed_queue
    assert node._action_queue.qsize() == 0
    assert node._vla_seeded is False
    assert node._vla_action_source_monotonic is None
    assert node._vla_action_epoch is None
    assert node._rtc_guided_streak == 0
    assert node._latency_tracker.values == []

    # With the stale leftovers gone, the next fresh result is necessarily a
    # new unguided seed rather than guidance from the old observation.
    assert not commit_vla_result(node, source=40.0, completed=40.5, elapsed=0.5)
    assert node._vla_seeded is True
    assert node._action_queue.qsize() == 35
    assert node._latency_tracker.values == [0.5]


def test_11829ms_regression_never_opens_readiness_gate() -> None:
    """The measured second-shadow latency cannot sustain chunk50 at 30 Hz."""
    node = make_vla_result_node()
    assert not commit_vla_result(
        node,
        source=10.0,
        completed=12.679,
        elapsed=2.6359,
    )
    assert not commit_vla_result(
        node,
        source=20.0,
        completed=21.240,
        elapsed=1.1829,
    )
    assert node._vla_seeded is True
    # Merge alignment is request→merge (36 steps), while the independent
    # observation→merge age remains 1.240 s for the action-age budget.
    assert node._action_queue.merge_calls[-1][2:] == (36, None)
    assert node._action_queue.qsize() == 14
    assert node._vla_policy_ready is False

    for index in range(5):
        source = 30.0 + index * 2.0
        assert not commit_vla_result(
            node,
            source=source,
            completed=source + 1.240,
            elapsed=1.1829,
        )

    assert node._watchdog.publish_allowed is True
    assert node._watchdog.latched is False
    assert node._rtc_guided_streak == 0
    assert node._vla_policy_ready is False
    assert any(
        "refill coverage" in message and "projected source age" in message
        for message in node.get_logger().warn_messages
    )


def test_rtc_merge_alignment_uses_request_to_merge_runtime_pre_ready() -> None:
    alignment = LeRobotInferenceNode._resolve_rtc_merge_alignment(
        queue_identity_matches=True,
        queue_size_before_inference=35,
        queue_size_at_merge=35,
        action_index_before_inference=0,
        action_index_at_merge=0,
        requested_at_monotonic=10.0,
        merge_at_monotonic=10.55,
        control_freq=30.0,
        policy_ready=False,
        index_phase_tolerance_steps=1,
    )

    assert alignment.runtime_sec == pytest.approx(0.55)
    assert alignment.wall_delay_steps == 17
    assert alignment.consumed_steps == 0
    assert alignment.merge_delay_steps == 17


def test_rtc_merge_alignment_uses_real_consumption_within_phase_bound() -> None:
    alignment = LeRobotInferenceNode._resolve_rtc_merge_alignment(
        queue_identity_matches=True,
        queue_size_before_inference=35,
        queue_size_at_merge=19,
        action_index_before_inference=4,
        action_index_at_merge=20,
        requested_at_monotonic=10.0,
        merge_at_monotonic=10.5,
        control_freq=30.0,
        policy_ready=True,
        index_phase_tolerance_steps=1,
    )

    assert alignment.wall_delay_steps == 15
    assert alignment.consumed_steps == 16
    assert alignment.merge_delay_steps == 16


def test_rtc_merge_alignment_accepts_scheduler_delayed_consumption() -> None:
    alignment = LeRobotInferenceNode._resolve_rtc_merge_alignment(
        queue_identity_matches=True,
        queue_size_before_inference=35,
        queue_size_at_merge=25,
        action_index_before_inference=4,
        action_index_at_merge=14,
        requested_at_monotonic=10.0,
        merge_at_monotonic=10.467536,
        control_freq=30.0,
        policy_ready=True,
        index_phase_tolerance_steps=1,
    )

    assert alignment.wall_delay_steps == 15
    assert alignment.consumed_steps == 10
    assert alignment.merge_delay_steps == 10


def test_rtc_merge_alignment_rejects_consumption_above_upper_guard() -> None:
    with pytest.raises(ValueError, match="exceeds wall-clock upper bound"):
        LeRobotInferenceNode._resolve_rtc_merge_alignment(
            queue_identity_matches=True,
            queue_size_before_inference=35,
            queue_size_at_merge=18,
            action_index_before_inference=4,
            action_index_at_merge=21,
            requested_at_monotonic=10.0,
            merge_at_monotonic=10.5,
            control_freq=30.0,
            policy_ready=True,
            index_phase_tolerance_steps=1,
        )


def test_rtc_merge_alignment_rejects_replaced_or_incoherent_queue() -> None:
    common = {
        "queue_size_before_inference": 35,
        "queue_size_at_merge": 20,
        "action_index_before_inference": 0,
        "action_index_at_merge": 15,
        "requested_at_monotonic": 10.0,
        "merge_at_monotonic": 10.5,
        "control_freq": 30.0,
        "policy_ready": True,
        "index_phase_tolerance_steps": 1,
    }
    with pytest.raises(ValueError, match="queue changed"):
        LeRobotInferenceNode._resolve_rtc_merge_alignment(
            queue_identity_matches=False,
            **common,
        )
    with pytest.raises(ValueError, match="depth/index consumption mismatch"):
        LeRobotInferenceNode._resolve_rtc_merge_alignment(
            queue_identity_matches=True,
            queue_size_at_merge=21,
            **{
                key: value
                for key, value in common.items()
                if key != "queue_size_at_merge"
            },
        )


def test_real_lerobot_action_queue_merges_at_consumed_index() -> None:
    torch = pytest.importorskip("torch")
    action_queue_module = pytest.importorskip("lerobot.policies.rtc.action_queue")
    rtc_config_module = pytest.importorskip("lerobot.policies.rtc.configuration_rtc")
    queue = action_queue_module.ActionQueue(
        rtc_config_module.RTCConfig(enabled=True)
    )
    initial = torch.arange(50, dtype=torch.float32).reshape(50, 1)
    queue.merge(initial, initial, 0, None)

    node = make_vla_result_node()
    node._action_queue = queue
    node._vla_warmup_pending = False
    node._vla_policy_ready = True
    node._vla_seeded = True
    with node._action_queue_lock, patch(
        "lerobot_control.inference_node.time.monotonic", return_value=10.0
    ):
        dispatch, leftover = node._capture_rtc_dispatch_locked()
    assert dispatch.queue_size == len(leftover) == 50

    for _ in range(15):
        assert queue.get() is not None

    replacement = torch.arange(100, 150, dtype=torch.float32).reshape(50, 1)
    with node._safety_lock, patch(
        "lerobot_control.inference_node.time.monotonic", return_value=10.5
    ):
        queue_size, alignment = node._merge_vla_result_locked(
            original=replacement,
            processed=replacement,
            dispatch=dispatch,
            observation_monotonic=10.0,
            epoch=0,
        )

    assert alignment.runtime_sec == pytest.approx(0.5)
    assert alignment.wall_delay_steps == 15
    assert alignment.consumed_steps == alignment.merge_delay_steps == 15
    assert queue_size == queue.qsize() == 35
    assert queue.get_action_index() == 0
    assert torch.equal(queue.get_left_over(), replacement[15:])


def test_pre_ready_queue_threshold_never_waits_for_an_unpublished_queue() -> None:
    assert not LeRobotInferenceNode._rtc_should_wait_for_refill(
        queue_size=35,
        queue_threshold=20,
        policy_ready=False,
    )
    assert LeRobotInferenceNode._rtc_should_wait_for_refill(
        queue_size=35,
        queue_threshold=20,
        policy_ready=True,
    )


def test_guided_delay_window_supersedes_slow_provisional_seed() -> None:
    assert LeRobotInferenceNode._rtc_inference_delay_steps(
        guided_latencies_sec=(),
        tracked_max_latency_sec=1.1829,
        control_freq=30.0,
        fallback_steps=10,
    ) == 36
    assert LeRobotInferenceNode._rtc_inference_delay_steps(
        guided_latencies_sec=(0.55,) * 5,
        tracked_max_latency_sec=1.1829,
        control_freq=30.0,
        fallback_steps=10,
    ) == 17
    assert LeRobotInferenceNode._rtc_inference_delay_steps(
        guided_latencies_sec=(),
        tracked_max_latency_sec=0.0,
        control_freq=30.0,
        fallback_steps=10,
    ) == 10


def test_rtc_readiness_exact_coverage_boundary() -> None:
    common = {
        "candidate_delay_steps": 15,
        "guided_latencies_sec": (0.5,),
        "control_freq": 30.0,
        "queue_threshold": 50,
        "source_age_sec": 0.5,
        "max_action_age_sec": 1.5,
        "execution_horizon": 20,
        "latency_guard_steps": 2,
        "scheduler_guard_steps": 1,
        "min_guided_overlap_steps": 0,
    }

    exact = LeRobotInferenceNode._assess_rtc_readiness(
        chunk_size=34,
        **common,
    )
    short = LeRobotInferenceNode._assess_rtc_readiness(
        chunk_size=33,
        **common,
    )

    assert exact.q_trigger == 19
    assert exact.q_required == exact.coverage_required_steps == 18
    assert exact.sustainable is True
    assert short.q_required == 17
    assert short.sustainable is False
    assert any("refill coverage" in reason for reason in short.failures)


def test_rtc_readiness_source_age_boundary_is_strict() -> None:
    common = {
        "chunk_size": 50,
        "candidate_delay_steps": 15,
        "guided_latencies_sec": (0.5,),
        "control_freq": 30.0,
        "queue_threshold": 22,
        "source_age_sec": 0.5,
        "execution_horizon": 20,
        "latency_guard_steps": 2,
        "scheduler_guard_steps": 1,
        "min_guided_overlap_steps": 0,
    }
    reference = LeRobotInferenceNode._assess_rtc_readiness(
        max_action_age_sec=10.0,
        **common,
    )
    exact = LeRobotInferenceNode._assess_rtc_readiness(
        max_action_age_sec=reference.age_at_next_refill_sec,
        **common,
    )
    margin = LeRobotInferenceNode._assess_rtc_readiness(
        max_action_age_sec=reference.age_at_next_refill_sec + 1e-9,
        **common,
    )

    assert exact.age_at_next_refill_sec == pytest.approx(1.5333333333333332)
    assert exact.sustainable is False
    assert any("projected source age" in reason for reason in exact.failures)
    assert margin.sustainable is True


def test_rtc_readiness_useful_overlap_boundary() -> None:
    common = {
        "chunk_size": 50,
        "candidate_delay_steps": 15,
        "guided_latencies_sec": (0.5,),
        "control_freq": 30.0,
        "queue_threshold": 50,
        "source_age_sec": 0.5,
        "max_action_age_sec": 1.5,
        "latency_guard_steps": 2,
        "scheduler_guard_steps": 1,
        "min_guided_overlap_steps": 3,
    }
    exact = LeRobotInferenceNode._assess_rtc_readiness(
        execution_horizon=20,
        **common,
    )
    short = LeRobotInferenceNode._assess_rtc_readiness(
        execution_horizon=19,
        **common,
    )

    assert exact.useful_guided_overlap_steps == 3
    assert exact.sustainable is True
    assert short.useful_guided_overlap_steps == 2
    assert short.sustainable is False
    assert any("useful guided overlap" in reason for reason in short.failures)


def test_current_horizon_cannot_open_gate_at_steady_550ms() -> None:
    assessment = LeRobotInferenceNode._assess_rtc_readiness(
        chunk_size=50,
        candidate_delay_steps=17,
        guided_latencies_sec=(0.55,) * 5,
        control_freq=30.0,
        queue_threshold=50,
        source_age_sec=0.55,
        max_action_age_sec=1.5,
        execution_horizon=12,
        latency_guard_steps=2,
        scheduler_guard_steps=1,
        min_guided_overlap_steps=3,
    )

    assert assessment.q_start == 33
    assert assessment.coverage_required_steps == 20
    assert assessment.age_at_next_refill_sec < 1.5
    assert assessment.useful_guided_overlap_steps == 0
    assert assessment.sustainable is False


def test_failed_guided_refill_resets_consecutive_readiness_streak() -> None:
    node = make_vla_result_node()
    assert not commit_vla_result(node, source=10.0, completed=12.6, elapsed=2.6)
    assert not commit_vla_result(node, source=20.0, completed=20.55, elapsed=0.55)
    assert not commit_vla_result(node, source=21.0, completed=21.55, elapsed=0.55)
    assert not commit_vla_result(node, source=22.0, completed=22.55, elapsed=0.55)
    assert node._rtc_guided_streak == 2

    assert not commit_vla_result(
        node,
        source=30.0,
        completed=31.24,
        elapsed=1.1829,
    )
    assert node._rtc_guided_streak == 0
    assert node._vla_policy_ready is False
    assert node._vla_seeded is False
    assert node._action_queue.qsize() == 0

    # The failed guided candidate is not retained. First create a new unguided
    # seed, then prove five consecutive guided refills from that seed.
    for index in range(5):
        source = 40.0 + index
        assert not commit_vla_result(node, source=source, completed=source + 0.55)
    assert commit_vla_result(node, source=50.0, completed=50.55)
    assert node._vla_policy_ready is True


def test_post_ready_sustainability_loss_latches_and_clears_before_publish() -> None:
    node = make_vla_result_node()
    assert not commit_vla_result(node, source=10.0, completed=12.6, elapsed=2.6)
    assert not commit_vla_result(node, source=20.0, completed=20.55, elapsed=0.55)
    for index in range(4):
        source = 30.0 + index
        assert not commit_vla_result(node, source=source, completed=source + 0.55)
    assert commit_vla_result(node, source=40.0, completed=40.55)
    assert node._vla_policy_ready is True
    policy_epoch = node._policy_epoch

    assert not commit_vla_result(
        node,
        source=50.0,
        completed=51.24,
        elapsed=1.1829,
    )

    assert node._watchdog.latched is True
    assert node._watchdog.publish_allowed is False
    assert "RTC sustainability lost" in node._watchdog.trip_reason
    assert node._policy_epoch == policy_epoch + 1
    assert node._vla_policy_ready is False
    assert node._vla_seeded is False
    assert node._action_queue.qsize() == 0


def test_post_ready_merge_uses_real_consumption_when_it_exceeds_wall_delay() -> None:
    node = make_vla_result_node()
    node._vla_warmup_pending = False
    node._vla_policy_ready = True
    node._vla_seeded = True
    node._rtc_guided_streak = 5
    node._rtc_guided_latencies.extend([0.5] * 5)
    node._action_queue = FakeVlaQueue([np.array([0.0])] * 35)
    dispatch = RTCDispatchSnapshot(
        queue=node._action_queue,
        queue_size=node._action_queue.qsize(),
        action_index=node._action_queue.get_action_index(),
        requested_at_monotonic=10.0,
    )
    for _ in range(16):
        assert node._action_queue.get() is not None

    assert commit_vla_result(
        node,
        source=10.0,
        completed=10.5,
        elapsed=0.5,
        guided=True,
        dispatch=dispatch,
    )

    assert node._watchdog.publish_allowed is True
    assert node._action_queue.merge_calls[-1][2:] == (16, None)
    assert node._action_queue.qsize() == 34


def test_post_ready_excess_consumption_latches_before_merge() -> None:
    node = make_vla_result_node()
    node._vla_warmup_pending = False
    node._vla_policy_ready = True
    node._vla_seeded = True
    node._rtc_guided_streak = 5
    node._rtc_guided_latencies.extend([0.5] * 5)
    old_queue = FakeVlaQueue([np.array([0.0])] * 35)
    node._action_queue = old_queue
    dispatch = RTCDispatchSnapshot(
        queue=old_queue,
        queue_size=old_queue.qsize(),
        action_index=old_queue.get_action_index(),
        requested_at_monotonic=10.0,
    )
    for _ in range(17):
        assert old_queue.get() is not None

    assert not commit_vla_result(
        node,
        source=10.0,
        completed=10.5,
        elapsed=0.5,
        guided=True,
        dispatch=dispatch,
    )

    assert old_queue.merge_calls == []
    assert node._watchdog.latched is True
    assert node._watchdog.publish_allowed is False
    assert "exceeds wall-clock upper bound" in node._watchdog.trip_reason
    assert node._vla_policy_ready is False
    assert node._action_queue.qsize() == 0


def test_vla_publish_loop_does_not_touch_queue_before_policy_ready() -> None:
    node = make_vla_result_node()
    node._shutting_down = False
    node._evaluate_watchdog = lambda: True
    node._debug = False
    node._queue_depths = deque()
    node._vla_skip_count = 0
    node._action_queue.actions.append(np.array([0.1, 0.2]))
    node._publish_action = lambda _action: pytest.fail(
        "pre-ready action must not be published"
    )

    node._publish_loop()

    assert node._action_queue.get_calls == 0
    assert node.metrics.control_calls == 1


def test_empty_queue_after_policy_ready_latches_instead_of_skipping() -> None:
    node = make_vla_result_node()
    node._shutting_down = False
    node._evaluate_watchdog = lambda: True
    node._debug = False
    node._queue_depths = deque()
    node._vla_skip_count = 0
    node._vla_policy_ready = True
    node._vla_seeded = True
    old_queue = node._action_queue
    node._publish_action = lambda _action: pytest.fail(
        "an empty RTC queue must never publish"
    )

    node._publish_loop()

    assert old_queue.get_calls == 1
    assert node._vla_skip_count == 1
    assert node._watchdog.latched is True
    assert node._watchdog.publish_allowed is False
    assert "queue emptied" in node._watchdog.trip_reason
    assert node._vla_policy_ready is False
    assert node._action_queue.qsize() == 0


def test_vla_invalidation_restores_warmup_and_not_ready_state() -> None:
    node = make_vla_result_node()
    node._vla_warmup_pending = False
    node._vla_policy_ready = True
    node._vla_stale_result_count = 4
    node._classic_action_deque = deque()
    node._classic_chunk_source_monotonic = None
    node._delta_ref_state = None
    node._abs_shadow_queue = deque()
    node._model_lock = threading.Lock()
    node._obs_lock = threading.Lock()
    node._latest_obs = "observation"
    node._last_inferred_observation_sequence = "sequence"
    node.model = SimpleNamespace()
    node._new_vla_action_queue = FakeVlaQueue

    node._invalidate_action_state_locked()

    assert node._policy_epoch == 1
    assert node._vla_warmup_pending is True
    assert node._vla_policy_ready is False
    assert node._vla_stale_result_count == 0
    assert node._vla_seeded is False
    assert node._rtc_guided_streak == 0
    assert list(node._rtc_guided_latencies) == []
    assert node._vla_action_source_monotonic is None
    assert node._vla_action_epoch is None
    assert node._latency_tracker.reset_calls == 1
    assert node._latest_obs is None
    assert node._last_inferred_observation_sequence is None
    assert isinstance(node._action_queue, FakeVlaQueue)


def test_reset_rejects_a_concurrent_pre_reset_forward_without_changing_watchdog() -> None:
    """An old forward cannot cross reset and consume the new warm-up slot."""
    node = make_vla_result_node()
    node._vla_warmup_pending = False
    node._vla_policy_ready = True
    node._classic_action_deque = deque()
    node._classic_chunk_source_monotonic = None
    node._delta_ref_state = np.array([0.0])
    node._abs_shadow_queue = deque([np.array([0.1])])
    node._obs_lock = threading.Lock()
    node._latest_obs = "pre-reset observation"
    node._last_inferred_observation_sequence = "pre-reset sequence"
    node._new_vla_action_queue = FakeVlaQueue

    reset_has_invalidated = threading.Event()
    forward_holds_model = threading.Event()
    release_forward = threading.Event()
    model_reset_started = threading.Event()
    release_model_reset = threading.Event()
    old_commit_done = threading.Event()
    old_commit_result = []

    class SignallingLimiter(FakeLimiter):
        def reset(self) -> None:
            super().reset()
            reset_has_invalidated.set()

    class BlockingResetModel:
        config = SimpleNamespace(
            rtc_config=SimpleNamespace(execution_horizon=24)
        )

        def reset(self) -> None:
            model_reset_started.set()
            release_model_reset.wait(timeout=2.0)

    node.action_limiter = SignallingLimiter()
    node.model = BlockingResetModel()
    pre_reset_policy_epoch = node._policy_epoch

    def finish_old_forward() -> None:
        with node._model_lock:
            forward_holds_model.set()
            release_forward.wait(timeout=2.0)
        old_commit_result.append(
            commit_vla_result(
                node,
                source=10.0,
                completed=10.5,
                policy_epoch=pre_reset_policy_epoch,
            )
        )
        old_commit_done.set()

    forward_thread = threading.Thread(target=finish_old_forward)
    reset_thread = threading.Thread(target=node.reset_policy)
    forward_thread.start()
    assert forward_holds_model.wait(timeout=1.0)
    reset_thread.start()

    # reset_policy owns the safety lock and has advanced the policy epoch, but
    # it must wait for the already-running forward to release the model lock.
    assert reset_has_invalidated.wait(timeout=1.0)
    assert node._policy_epoch == pre_reset_policy_epoch + 1
    release_forward.set()
    assert model_reset_started.wait(timeout=1.0)
    assert not old_commit_done.is_set()

    release_model_reset.set()
    forward_thread.join(timeout=1.0)
    reset_thread.join(timeout=1.0)
    assert not forward_thread.is_alive()
    assert not reset_thread.is_alive()

    assert old_commit_result == [False]
    assert node._watchdog.epoch == 0
    assert node._watchdog.publish_allowed is True
    assert node._watchdog.latched is False
    assert node._vla_warmup_pending is True
    assert node._vla_policy_ready is False
    assert node._vla_action_source_monotonic is None
    assert node._vla_action_epoch is None
    assert node._latest_obs is None
    assert node._last_inferred_observation_sequence is None
    assert node._action_queue.merge_calls == []

    # The first new-epoch result remains the cold discard. The next result is a
    # provisional seed, followed by the full five-guided-refill proof.
    assert not commit_vla_result(node, source=20.0, completed=20.5)
    assert node._vla_warmup_pending is False
    assert node._vla_policy_ready is False
    assert not commit_vla_result(node, source=21.0, completed=21.5)
    assert node._vla_seeded is True
    for index in range(4):
        source = 22.0 + index
        assert not commit_vla_result(node, source=source, completed=source + 0.5)
    assert commit_vla_result(node, source=30.0, completed=30.5)
    assert node._vla_policy_ready is True


def test_successful_rearm_restarts_full_rtc_readiness_proof() -> None:
    node = make_vla_result_node()
    node.camera_names = []
    node.strategy = SimpleNamespace(get_input_snapshot=lambda _cameras: object())
    node._vla_warmup_pending = False
    node._vla_policy_ready = True
    node._vla_seeded = True
    node._rtc_guided_streak = 5
    node._rtc_guided_latencies.extend([0.5] * 5)
    node._action_queue.actions.extend([np.array([0.1])] * 10)
    node._watchdog.trip("test fault")
    old_policy_epoch = node._policy_epoch
    old_watchdog_epoch = node._watchdog.epoch
    response = SimpleNamespace(success=None, message=None)

    result = node._handle_watchdog_rearm(None, response)

    assert result.success is True
    assert node._watchdog.publish_allowed is True
    assert node._watchdog.epoch == old_watchdog_epoch + 1
    assert node._policy_epoch == old_policy_epoch + 1
    assert node._vla_warmup_pending is True
    assert node._vla_policy_ready is False
    assert node._vla_seeded is False
    assert node._rtc_guided_streak == 0
    assert list(node._rtc_guided_latencies) == []
    assert node._action_queue.qsize() == 0

    assert not commit_vla_result(node, source=20.0, completed=20.5)
    assert not commit_vla_result(node, source=21.0, completed=21.5)
    for index in range(4):
        source = 22.0 + index
        assert not commit_vla_result(node, source=source, completed=source + 0.5)
    assert commit_vla_result(node, source=30.0, completed=30.5)
    assert node._vla_policy_ready is True


def test_expired_queued_vla_action_still_latches_after_policy_ready() -> None:
    node = make_vla_result_node()
    node._shutting_down = False
    node._evaluate_watchdog = lambda: True
    node._vla_policy_ready = True
    node._vla_action_source_monotonic = 10.0
    node._vla_action_epoch = 0
    node._action_queue.actions.append(np.array([0.1, 0.2]))
    node._debug = False
    node._queue_depths = deque()
    node._vla_skip_count = 0
    node.camera_names = []
    node.strategy = SimpleNamespace(get_input_snapshot=lambda _cameras: None)
    publish_calls = []
    node._publish_action = publish_calls.append
    captured = []
    node._apply_watchdog_result_locked = captured.append

    def expire_action(**_kwargs):
        return node._watchdog.trip("action source observation is too old")

    node._watchdog.authorize_action = expire_action

    node._publish_loop()

    assert node._watchdog.publish_allowed is False
    assert node._watchdog.latched is True
    assert captured[-1].state is WatchdogState.LATCHED
    assert publish_calls == []


def test_fault_invalidation_clears_every_classic_action_buffer() -> None:
    node = object.__new__(LeRobotInferenceNode)
    node.model_type = "act"
    node._classic_action_deque = deque([np.array([1.0])])
    node._classic_chunk_source_monotonic = 10.0
    node._vla_action_source_monotonic = 10.0
    node._vla_action_epoch = 0
    node._delta_ref_state = np.array([1.0])
    node._abs_shadow_queue = deque([np.array([2.0])])
    node.action_limiter = FakeLimiter()
    node._model_lock = threading.Lock()
    internal_action_queue = FakeQueue()
    internal_named_queue = FakeQueue()
    node.model = SimpleNamespace(
        _action_queue=internal_action_queue,
        _queues={"action": internal_named_queue},
    )

    node._invalidate_action_state_locked()

    assert not node._classic_action_deque
    assert node._delta_ref_state is None
    assert not node._abs_shadow_queue
    assert node.action_limiter.reset_calls == 1
    assert internal_action_queue.items == []
    assert internal_named_queue.items == []


def test_publish_loop_is_suppressed_before_touching_the_queue_when_watchdog_denies() -> None:
    node = object.__new__(LeRobotInferenceNode)
    node._shutting_down = False
    node._evaluate_watchdog = lambda: False
    node.metrics = FakeMetrics()
    node._classic_action_deque = deque([np.array([1.0])])

    node._publish_loop()

    assert node.metrics.control_calls == 0
    assert len(node._classic_action_deque) == 1


def test_watchdog_snapshot_runtime_error_latches_and_invalidates_queues() -> None:
    node = make_timer_safety_node()

    def raise_snapshot_error(_camera_names):
        raise RuntimeError("shared image snapshot did not stabilize")

    node.strategy = SimpleNamespace(get_input_snapshot=raise_snapshot_error)

    assert node._evaluate_watchdog() is False
    assert node._watchdog.publish_allowed is False
    assert "shared image snapshot did not stabilize" in node._watchdog.trip_reason
    assert not node._classic_action_deque
    assert node._delta_ref_state is None
    assert not node._abs_shadow_queue
    assert node.action_limiter.reset_calls == 1


def test_rearm_snapshot_runtime_error_is_caught_and_latches_fail_closed() -> None:
    node = make_timer_safety_node()

    def raise_snapshot_error(_camera_names):
        raise RuntimeError("shared rearm snapshot did not stabilize")

    node.strategy = SimpleNamespace(get_input_snapshot=raise_snapshot_error)
    response = SimpleNamespace(success=None, message=None)

    result = node._handle_watchdog_rearm(None, response)

    assert result is response
    assert response.success is False
    assert "shared rearm snapshot did not stabilize" in response.message
    assert node._watchdog.publish_allowed is False
    assert "shared rearm snapshot did not stabilize" in node._watchdog.trip_reason
    assert not node._classic_action_deque
    assert node._delta_ref_state is None
    assert not node._abs_shadow_queue
    assert node.action_limiter.reset_calls == 1


def test_rearm_snapshot_runtime_error_keeps_an_existing_latch_closed() -> None:
    node = make_timer_safety_node()
    node._watchdog.trip("existing input fault")
    node._classic_action_deque.clear()
    node._delta_ref_state = None
    node._abs_shadow_queue.clear()

    def raise_snapshot_error(_camera_names):
        raise RuntimeError("shared rearm snapshot did not stabilize")

    node.strategy = SimpleNamespace(get_input_snapshot=raise_snapshot_error)
    response = SimpleNamespace(success=None, message=None)

    result = node._handle_watchdog_rearm(None, response)

    assert result is response
    assert response.success is False
    assert node._watchdog.publish_allowed is False
    assert "shared rearm snapshot did not stabilize" in node._watchdog.trip_reason
    assert node.action_limiter.reset_calls == 0


def test_observation_snapshot_runtime_error_latches_before_inference() -> None:
    node = make_timer_safety_node()
    inference_calls = []

    def raise_observation_error(_camera_names):
        raise RuntimeError("shared observation snapshot did not stabilize")

    node._evaluate_watchdog = lambda: True
    node.strategy = SimpleNamespace(
        get_observation=raise_observation_error,
        get_input_snapshot=lambda _camera_names: None,
    )
    node._preprocess_vla_observation = lambda observation: inference_calls.append(
        observation
    )

    node._obs_update()

    assert node._watchdog.publish_allowed is False
    assert "shared observation snapshot did not stabilize" in node._watchdog.trip_reason
    assert not node._classic_action_deque
    assert inference_calls == []


def test_publish_snapshot_runtime_error_latches_without_publishing() -> None:
    node = make_timer_safety_node()
    publish_calls = []

    def raise_snapshot_error(_camera_names):
        raise RuntimeError("shared authorization snapshot did not stabilize")

    node._evaluate_watchdog = lambda: True
    node.strategy = SimpleNamespace(get_input_snapshot=raise_snapshot_error)
    node._publish_action = lambda action: publish_calls.append(action)

    node._publish_loop()

    assert node._watchdog.publish_allowed is False
    assert "shared authorization snapshot did not stabilize" in node._watchdog.trip_reason
    assert not node._classic_action_deque
    assert publish_calls == []


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (np.array([1.0]), "invalid action dimension"),
        (np.array([np.nan, 0.0]), "non-finite"),
    ],
)
def test_invalid_action_is_rejected_before_any_publish(action, expected) -> None:
    node = object.__new__(LeRobotInferenceNode)
    node.arms_config = {
        "left": {
            "action_start": 0,
            "action_end": 2,
            "ros_prefix": "follower_l",
        }
    }
    node.joint_names_config = {"controller_joint_order": ["joint1", "joint2"]}
    node.strategy = SimpleNamespace(
        get_current_joint_positions=lambda: {
            "follower_l_joint1": 0.0,
            "follower_l_joint2": 0.0,
        }
    )
    node._joint_position_limits = {
        "follower_l_joint1": (-1.0, 1.0),
        "follower_l_joint2": (-1.0, 1.0),
    }
    node._joint_limit_tolerance = 1e-6

    with pytest.raises(ValueError, match=expected):
        node._publish_action(action)


def test_missing_current_joint_is_never_replaced_with_zero() -> None:
    node = object.__new__(LeRobotInferenceNode)
    node.arms_config = {
        "left": {
            "action_start": 0,
            "action_end": 2,
            "ros_prefix": "follower_l",
        }
    }
    node.joint_names_config = {"controller_joint_order": ["joint1", "joint2"]}
    node.strategy = SimpleNamespace(
        get_current_joint_positions=lambda: {"follower_l_joint1": 0.0}
    )
    node._joint_position_limits = {
        "follower_l_joint1": (-1.0, 1.0),
        "follower_l_joint2": (-1.0, 1.0),
    }
    node._joint_limit_tolerance = 1e-6

    with pytest.raises(ValueError, match="joint positions missing"):
        node._publish_action(np.array([0.0, 0.0]))


def test_partial_joint_limit_mapping_rejected_at_configuration() -> None:
    node = make_two_arm_node()

    with pytest.raises(ValueError, match="exactly cover"):
        node._parse_joint_position_limits(
            {
                "follower_l_joint1": [-1.0, 1.0],
                "follower_l_joint2": [-1.0, 1.0],
            }
        )


@pytest.mark.parametrize(
    "invalid_tolerance",
    [float("nan"), float("inf"), -1e-9, 1.000001e-6],
)
def test_invalid_joint_limit_tolerance_rejected(invalid_tolerance) -> None:
    with pytest.raises(ValueError, match="finite and between"):
        LeRobotInferenceNode._parse_joint_limit_tolerance(invalid_tolerance)


def test_duplicate_camera_feature_mapping_is_rejected() -> None:
    node = object.__new__(LeRobotInferenceNode)
    parameters = {
        "model_path": "",
        "config_file": "unused.yaml",
        "control_frequency": 30.0,
        "enforce_joint_position_limits": True,
        "device": "cpu",
        "echo_topic_only": True,
        "debug": False,
        "debug_image_dir": "",
        "monitor_enable": False,
        "joint_state_worker": False,
    }
    node.declare_parameter = lambda *_args, **_kwargs: None
    node.get_parameter = lambda name: SimpleNamespace(value=parameters[name])
    node._load_yaml_config = lambda _path: {
        "cameras": {
            "mapping": {
                "/camera/base/image": "base",
                "/camera/wrist/image": "base",
            }
        }
    }

    with pytest.raises(ValueError, match="unique.*base"):
        node._setup_config()


@pytest.mark.parametrize(
    "config_name",
    (
        "inference_default.yaml",
        "inference_default_afo.yaml",
        "inference_eval.yaml",
        "inference_single_arm.yaml",
    ),
)
def test_deploy_configs_have_complete_valid_urdf_limit_mapping(config_name) -> None:
    config_path = find_deploy_config(config_name)
    config = yaml.safe_load(config_path.read_text())
    node = object.__new__(LeRobotInferenceNode)
    node.arms_config = config["arms"]
    node.joint_names_config = config["joint_names"]

    parsed = node._parse_joint_position_limits(
        config["safety"]["joint_position_limits"]
    )

    expected_count = 8 * len(config["arms"])
    assert len(parsed) == expected_count
    if "left" in config["arms"]:
        assert parsed["follower_l_joint2"] == (-3.3161, 0.1745)
    assert parsed["follower_r_joint2"] == (-0.1745, 3.3161)
    assert config["safety"]["joint_limit_tolerance"] == 1e-6
    assert config["watchdog"]["max_action_age_sec"] == 1.5
    rtc = config["inference_tuning"]["rtc"]
    assert rtc["readiness_guided_forwards"] == 5
    assert rtc["readiness_latency_guard_steps"] == 2
    assert rtc["readiness_index_phase_tolerance_steps"] == 1
    assert rtc["readiness_scheduler_guard_steps"] == 1
    assert rtc["readiness_min_guided_overlap_steps"] == 3


def test_envelope_profile_has_bounded_saturation_and_provenance() -> None:
    config_path = find_deploy_config("inference_envelope_afo.yaml")
    config = yaml.safe_load(config_path.read_text())
    node = object.__new__(LeRobotInferenceNode)
    node.arms_config = config["arms"]
    node.joint_names_config = config["joint_names"]
    node._joint_position_limits = node._parse_joint_position_limits(
        config["safety"]["joint_position_limits"]
    )
    node._saturate_joint_targets = node._parse_saturate_joint_targets(
        config["safety"]["saturate_joint_targets"]
    )

    margins = node._parse_saturate_joint_margins(
        config["safety"]["saturate_joint_margins"]
    )

    assert set(margins) == set(node._joint_position_limits)
    assert max(margins.values()) <= node.MAX_SATURATION_MARGIN_RAD
    assert config["diagnostics"] == {
        "rtc_timing": True,
        "rtc_cuda_events": False,
        "rtc_provenance": True,
    }
    assert config["inference_tuning"]["rtc"]["execution_horizon"] == 35
    assert config["watchdog"]["max_action_age_sec"] == 1.65


def test_bounded_saturation_clamps_inside_margin_and_counts() -> None:
    node = make_two_arm_node()
    node._saturate_joint_targets = frozenset({"follower_r_joint2"})
    node._saturate_joint_margins = {"follower_r_joint2": 0.02}

    node._publish_action(np.array([0.1, 0.2, 0.3, 1.015]))

    assert node.arm_publishers["left"].messages == [[0.1, 0.2]]
    assert node.arm_publishers["right"].messages == [[0.3, 1.0]]
    assert node._saturation_counts == {"follower_r_joint2": 1}
    assert len(node.get_logger().warn_messages) == 1


def test_saturation_past_margin_still_fails_closed() -> None:
    node = make_two_arm_node()
    node._saturate_joint_targets = frozenset({"follower_r_joint2"})
    node._saturate_joint_margins = {"follower_r_joint2": 0.02}

    with pytest.raises(ValueError, match="follower_r_joint2"):
        node._publish_action(np.array([0.1, 0.2, 0.3, 1.021]))

    assert node.arm_publishers["left"].messages == []
    assert node.arm_publishers["right"].messages == []
    assert node._saturation_counts == {}


def test_disabled_joint_ranges_publish_unclamped_target() -> None:
    node = make_two_arm_node()
    node._enforce_joint_position_limits = False
    node._saturate_joint_targets = frozenset({"follower_r_joint2"})
    node._saturate_joint_margins = {"follower_r_joint2": 0.02}

    node._publish_action(np.array([0.1, 0.2, 0.3, 1.5]))

    assert node.arm_publishers["left"].messages == [[0.1, 0.2]]
    assert node.arm_publishers["right"].messages == [[0.3, 1.5]]
    assert node._saturation_counts == {}


def test_disabled_joint_ranges_still_reject_non_finite_target() -> None:
    node = make_two_arm_node()
    node._enforce_joint_position_limits = False

    with pytest.raises(ValueError, match="non-finite"):
        node._validate_absolute_joint_targets(
            ["follower_l_joint1"],
            np.array([np.nan]),
            stage="raw absolute",
        )


@pytest.mark.parametrize("margin", [0.0, -0.01, 0.051, float("nan")])
def test_invalid_saturation_margin_is_rejected(margin) -> None:
    node = make_two_arm_node()
    node._saturate_joint_targets = frozenset({"follower_r_joint2"})

    with pytest.raises(ValueError, match="saturate_joint_margins"):
        node._parse_saturate_joint_margins({"follower_r_joint2": margin})


@pytest.mark.parametrize(
    "invalid_bounds",
    ([0.0, 0.0], [1.0, -1.0], [float("nan"), 1.0]),
)
def test_invalid_joint_bounds_rejected(invalid_bounds) -> None:
    node = make_two_arm_node()
    limits = {name: [-1.0, 1.0] for name in node._joint_position_limits}
    limits["follower_l_joint1"] = invalid_bounds

    with pytest.raises(ValueError, match="invalid joint limit"):
        node._parse_joint_position_limits(limits)


def test_controller_order_processing_preserves_normal_process_behavior() -> None:
    limiter = ActionLimiter(
        max_delta=0.04,
        model_joint_order=["finger_joint1", "joint1"],
        controller_joint_order=["joint1", "finger_joint1"],
    )
    model_order = np.array([0.05, 0.2])
    current = np.array([0.0, 0.049])

    normal = limiter.process(model_order, current)
    controller_order = limiter.reorder(model_order)
    explicit = limiter.process_controller_order(controller_order, current)

    np.testing.assert_allclose(normal, explicit)


def test_right_arm_limit_failure_publishes_neither_arm() -> None:
    node = make_two_arm_node()

    with pytest.raises(ValueError, match="follower_r_joint2"):
        node._publish_action(np.array([0.1, 0.2, 0.3, 1.01]))

    assert node.arm_publishers["left"].messages == []
    assert node.arm_publishers["right"].messages == []
    assert not node._has_published


def test_raw_impossible_target_is_not_hidden_by_real_delta_limiter() -> None:
    node = make_two_arm_node()
    node.action_limiter = ActionLimiter(
        max_delta=0.05,
        model_joint_order=["joint1", "joint2"],
        controller_joint_order=["joint1", "joint2"],
    )

    with pytest.raises(ValueError, match="raw absolute.*follower_r_joint2"):
        node._publish_action(np.array([0.1, 0.2, 0.3, 100.0]))

    assert node.arm_publishers["left"].messages == []
    assert node.arm_publishers["right"].messages == []
    assert not node._has_published


def test_publish_loop_latches_absolute_limit_failure() -> None:
    node = make_two_arm_node()
    node._shutting_down = False
    node._evaluate_watchdog = lambda: True
    node._safety_lock = threading.RLock()
    node.model_type = "act"
    node._watchdog = FakeArmedWatchdog()
    node.strategy.get_input_snapshot = lambda _cameras: None
    node.camera_names = []
    node._classic_action_deque = deque(
        [(np.array([0.1, 0.2, 0.3, 1.01]), 10.0, 0)]
    )
    captured = []
    node._apply_watchdog_result_locked = captured.append
    node.get_logger = lambda: FakeLogger()

    node._publish_loop()

    assert node._watchdog.publish_allowed is False
    assert "outside absolute limit" in node._watchdog.trip_reason
    assert captured[-1].state is WatchdogState.LATCHED
    assert node.arm_publishers["left"].messages == []
    assert node.arm_publishers["right"].messages == []


def test_joint_limit_boundary_with_tolerance_publishes_both_arms() -> None:
    node = make_two_arm_node()

    node._publish_action(np.array([-1.0000005, 1.0000005, -1.0, 1.0]))

    assert node.arm_publishers["left"].messages == [[-1.0, 1.0]]
    assert node.arm_publishers["right"].messages == [[-1.0, 1.0]]
    assert node._has_published


def test_rtc_timing_breakdown_is_correlated_and_does_not_change_runtime_state():
    node = object.__new__(LeRobotInferenceNode)
    node._rtc_timing_enabled = True
    logger = FakeLogger()
    node.get_logger = lambda: logger
    dispatch = RTCDispatchSnapshot(
        queue=object(),
        queue_size=50,
        action_index=0,
        requested_at_monotonic=20.0,
    )
    observation_timing = VLAObservationTiming(
        callback_started_monotonic=18.90,
        read_started_monotonic=19.00,
        read_completed_monotonic=19.01,
        preprocess_started_monotonic=19.02,
        ready_at_monotonic=19.05,
    )
    merge_timing = RTCMergeStageTiming(
        queue_lock_requested_at_monotonic=20.24,
        queue_lock_acquired_at_monotonic=20.25,
        alignment_merge_at_monotonic=20.25,
        merge_completed_at_monotonic=20.27,
    )

    node._log_rtc_pipeline_timing(
        sample_id=7,
        phase="readiness",
        guided=True,
        publish_ready_after=False,
        observation_monotonic=18.95,
        observation_timing=observation_timing,
        dispatch=dispatch,
        model_lock_requested_at=20.01,
        model_started_at=20.02,
        predict_started_at=20.03,
        predict_completed_at=20.20,
        postprocess_completed_at=20.21,
        safety_lock_requested_at=20.22,
        safety_lock_acquired_at=20.23,
        commit_completed_at=20.28,
        merge_timing=merge_timing,
    )

    assert len(logger.info_messages) == 1
    message = logger.info_messages[0]
    for expected in (
        "[RTC_TIMING] sample=7 phase=readiness guided=true publish_ready_after=false merged=true",
        "obs_read_ms=10.000",
        "obs_validate_ms=10.000",
        "obs_preprocess_ms=30.000",
        "model_lock_wait_ms=10.000",
        "model_setup_ms=10.000",
        "predict_ms=170.000",
        "postprocess_ms=10.000",
        "safety_lock_wait_ms=10.000",
        "queue_lock_wait_ms=10.000",
        "queue_merge_ms=20.000",
        "request_to_commit_ms=280.000",
        "request_to_merge_ms=250.000",
        "source_age_at_commit_ms=1330.000",
    ):
        assert expected in message


def test_rtc_timing_disabled_is_a_complete_noop():
    node = object.__new__(LeRobotInferenceNode)
    node._rtc_timing_enabled = False
    logger = FakeLogger()
    node.get_logger = lambda: logger

    node._log_rtc_pipeline_timing(
        sample_id=1,
        phase="seed",
        guided=False,
        publish_ready_after=False,
        observation_monotonic=1.0,
        observation_timing=None,
        dispatch=RTCDispatchSnapshot(object(), 0, 0, 1.0),
        model_lock_requested_at=1.0,
        model_started_at=1.0,
        predict_started_at=1.0,
        predict_completed_at=1.0,
        postprocess_completed_at=1.0,
        safety_lock_requested_at=1.0,
        safety_lock_acquired_at=1.0,
        commit_completed_at=1.0,
        merge_timing=RTCMergeStageTiming(),
    )

    assert logger.info_messages == []


def test_rtc_provenance_correlates_exact_sensor_samples_and_chunk() -> None:
    node = object.__new__(LeRobotInferenceNode)
    node._rtc_provenance_enabled = True
    logger = FakeLogger()
    node.get_logger = lambda: logger
    provenance = ObservationProvenance(
        joint_state=SensorReading("joint_states", 101, 19.97, 123.456),
        cameras=(
            SensorReading("camera:base", 201, 19.95, 123.450),
            SensorReading("camera:left_wrist", 202, 19.96, 123.451),
            SensorReading("camera:right_wrist", 203, 19.98, 123.452),
        ),
    )
    sequence = ObservationSequence(
        joint_state=101,
        cameras=(("base", 201), ("left_wrist", 202), ("right_wrist", 203)),
    )
    chunk = np.arange(12, dtype=np.float32).reshape(3, 4)

    node._log_rtc_provenance(
        sample_id=9,
        sequence=sequence,
        provenance=provenance,
        requested_at_monotonic=20.0,
        processed_chunk=chunk,
    )

    assert len(logger.info_messages) == 1
    message = logger.info_messages[0]
    for expected in (
        "[RTC_PROVENANCE] sample=9 observation_joint_seq=101",
        "oldest_receipt_age_ms=50.000",
        "receipt_skew_ms=30.000",
        "ros_stamp_skew_ms=6.000",
        "chunk_shape=3x4",
        "first_action=[0.000000,1.000000,2.000000,3.000000]",
        "joint_states_seq=101 joint_states_stamp=123.456000000",
        "camera_base_seq=201 camera_base_stamp=123.450000000",
    ):
        assert expected in message


def test_rtc_provenance_disabled_is_a_complete_noop() -> None:
    node = object.__new__(LeRobotInferenceNode)
    node._rtc_provenance_enabled = False
    logger = FakeLogger()
    node.get_logger = lambda: logger

    node._log_rtc_provenance(
        sample_id=1,
        sequence=ObservationSequence(1, ()),
        provenance=None,
        requested_at_monotonic=1.0,
        processed_chunk=np.zeros((1, 1), dtype=np.float32),
    )

    assert logger.info_messages == []
    assert logger.error_messages == []


def test_cuda_timing_waits_for_query_and_never_synchronizes():
    node = object.__new__(LeRobotInferenceNode)
    node._rtc_cuda_timing_enabled = True
    logger = FakeLogger()
    node.get_logger = lambda: logger
    start_event = FakeCudaEvent(ready=True, elapsed_ms=204.25)
    end_event = FakeCudaEvent(ready=False)
    node._rtc_pending_cuda_timings = deque(
        [PendingRTCCudaTiming(42, start_event, end_event)]
    )

    node._drain_rtc_cuda_timings()
    assert len(node._rtc_pending_cuda_timings) == 1
    assert start_event.elapsed_calls == 0
    assert logger.info_messages == []

    end_event.ready = True
    node._drain_rtc_cuda_timings()
    assert not node._rtc_pending_cuda_timings
    assert start_event.elapsed_calls == 1
    assert logger.info_messages == [
        "[RTC_CUDA_TIMING] sample=42 cuda_model_ms=204.250"
    ]


def test_cuda_timing_setup_failure_is_diagnostic_only():
    node = object.__new__(LeRobotInferenceNode)
    node._rtc_cuda_timing_enabled = True
    logger = FakeLogger()
    node.get_logger = lambda: logger

    with patch(
        "lerobot_control.inference_node.torch.cuda.Event",
        side_effect=RuntimeError("CUDA events unavailable"),
    ):
        assert node._start_rtc_cuda_timing() is None

    assert node._rtc_cuda_timing_enabled is False
    assert len(logger.warn_messages) == 1
    assert "disabled after event setup failed" in logger.warn_messages[0]


def test_merge_without_stage_timing_adds_no_diagnostic_clock_reads():
    node = make_vla_result_node()
    node._vla_warmup_pending = False
    dispatch = RTCDispatchSnapshot(
        queue=node._action_queue,
        queue_size=0,
        action_index=0,
        requested_at_monotonic=10.0,
    )
    original = np.zeros((50, 1), dtype=np.float32)

    with node._safety_lock, patch(
        "lerobot_control.inference_node.time.monotonic", return_value=10.5
    ) as monotonic:
        queue_size, alignment = node._merge_vla_result_locked(
            original=original,
            processed=original,
            dispatch=dispatch,
            observation_monotonic=10.0,
            epoch=0,
            stage_timing=None,
        )

    assert queue_size == 35
    assert alignment.runtime_sec == pytest.approx(0.5)
    assert monotonic.call_count == 1
