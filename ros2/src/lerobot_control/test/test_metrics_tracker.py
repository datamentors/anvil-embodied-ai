"""Metrics counter regression tests for shared-memory sensor workers."""

import pytest
from lerobot_control.metrics_tracker import MetricsTracker


def test_joint_state_batch_increment_is_exact_and_constant_work() -> None:
    metrics = MetricsTracker()

    metrics.record_joint_states(60_000)
    metrics.record_joint_state()

    assert metrics.get_stats()["joint_count"] == 60_001


def test_zero_joint_state_batch_does_not_start_metrics_clock() -> None:
    metrics = MetricsTracker()

    metrics.record_joint_states(0)

    assert metrics._start_time is None
    assert metrics.get_stats()["joint_count"] == 0


@pytest.mark.parametrize("count", [-1, -100])
def test_negative_joint_state_batch_is_rejected(count: int) -> None:
    metrics = MetricsTracker()

    with pytest.raises(ValueError, match=">= 0"):
        metrics.record_joint_states(count)


@pytest.mark.parametrize("count", [True, 1.5, "2", None])
def test_non_integer_joint_state_batch_is_rejected(count) -> None:
    metrics = MetricsTracker()

    with pytest.raises(TypeError, match="must be an integer"):
        metrics.record_joint_states(count)

