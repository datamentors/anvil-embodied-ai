"""Tests for causal command matching and the maximum command age.

Both are opt-in through DataConfig, so the first test here pins the historical
behaviour: with the defaults, nothing changes.

What the two options fix:

1. ``action_match="causal"`` — the default "nearest" search minimises
   ``abs(cmd_ts - obs_ts)``, so it can pick a command published *after* the
   observation. That leaks future information into the action.

2. ``action_max_age_s`` — without it, ``_last_known_action`` is never evicted,
   so a command from any time in the past can be held forward indefinitely.
"""

from collections import deque

import numpy as np

from mcap_converter.config.schema import ActionTopicConfig, DataConfig
from mcap_converter.core.extractor import BufferedStreamExtractor


def make_config(**kwargs) -> DataConfig:
    return DataConfig(
        action_topics={
            "/follower_l_forward_position_controller/commands": ActionTopicConfig(
                arm="left", joint_order=["joint1", "joint2"]
            ),
            "/follower_r_forward_position_controller/commands": ActionTopicConfig(
                arm="right", joint_order=["joint1", "joint2"]
            ),
        },
        **kwargs,
    )


def make_extractor(**kwargs) -> BufferedStreamExtractor:
    return BufferedStreamExtractor(
        config=make_config(**kwargs), buffer_seconds=5.0, fps=30, quiet=True
    )


def _buffer(ts_pos_pairs):
    buf = deque()
    for ts, pos in ts_pos_pairs:
        buf.append((ts, np.array(pos, dtype=np.float32), np.array([]), np.array([])))
    return buf


# --------------------------------------------------------------------------- #
# defaults


def test_defaults_are_the_historical_behaviour():
    cfg = DataConfig()
    assert cfg.action_match == "nearest"
    assert cfg.action_max_age_s is None


def test_default_nearest_can_select_a_command_from_the_future():
    """The behaviour causal mode exists to avoid, pinned so it stays visible."""
    ex = make_extractor()
    # 10 ms before the observation, and 1 ms after it. "nearest" prefers the
    # one that is closer in absolute time — the future one.
    buf = _buffer([(0.990, [1.0, 1.0]), (1.001, [9.0, 9.0])])
    pos, kind = ex._resolve_action_position("left", buf, 1.000, {})
    assert kind == "exact"
    np.testing.assert_array_equal(pos, [9.0, 9.0])


# --------------------------------------------------------------------------- #
# _find_last_at_or_before


def test_find_last_at_or_before_picks_the_newest_past_entry():
    ex = make_extractor()
    buf = _buffer([(0.90, [1.0, 1.0]), (0.95, [2.0, 2.0]), (1.05, [3.0, 3.0])])
    assert ex._find_last_at_or_before(buf, 1.00) == 1


def test_find_last_at_or_before_accepts_an_exact_timestamp_match():
    ex = make_extractor()
    buf = _buffer([(0.90, [1.0, 1.0]), (1.00, [2.0, 2.0]), (1.10, [3.0, 3.0])])
    assert ex._find_last_at_or_before(buf, 1.00) == 1


def test_find_last_at_or_before_returns_none_when_everything_is_newer():
    ex = make_extractor()
    buf = _buffer([(1.10, [1.0, 1.0]), (1.20, [2.0, 2.0])])
    assert ex._find_last_at_or_before(buf, 1.00) is None


def test_find_last_at_or_before_handles_an_empty_buffer():
    ex = make_extractor()
    assert ex._find_last_at_or_before(deque(), 1.00) is None


# --------------------------------------------------------------------------- #
# causal matching


def test_causal_never_reaches_forward():
    ex = make_extractor(action_match="causal")
    buf = _buffer([(0.990, [1.0, 1.0]), (1.001, [9.0, 9.0])])
    pos, kind = ex._resolve_action_position("left", buf, 1.000, {})
    assert kind == "exact"
    np.testing.assert_array_equal(pos, [1.0, 1.0])


def test_causal_falls_through_when_the_buffer_is_all_future():
    """No past command in the buffer, but one was published earlier: hold it."""
    ex = make_extractor(action_match="causal")
    ex._last_known_action["left"] = np.array([5.0, 5.0], dtype=np.float32)
    ex._last_known_action_ts["left"] = 0.95
    buf = _buffer([(1.10, [9.0, 9.0])])
    pos, kind = ex._resolve_action_position("left", buf, 1.00, {})
    assert kind == "hold_last"
    np.testing.assert_array_equal(pos, [5.0, 5.0])


# --------------------------------------------------------------------------- #
# maximum age


def test_max_age_rejects_a_stale_buffer_command_and_tallies_it():
    ex = make_extractor(action_match="causal", action_max_age_s=0.05)
    buf = _buffer([(0.80, [1.0, 1.0])])          # 200 ms old
    obs = {"left": {"pos": np.array([7.0, 7.0], dtype=np.float32)}}
    pos, kind = ex._resolve_action_position("left", buf, 1.00, obs)
    assert kind == "fallback_to_observation"
    np.testing.assert_array_equal(pos, [7.0, 7.0])
    assert ex.get_action_fill_stats()["left"]["stale"] == 1


def test_max_age_accepts_a_command_inside_the_window():
    ex = make_extractor(action_match="causal", action_max_age_s=0.05)
    buf = _buffer([(0.97, [1.0, 1.0])])          # 30 ms old
    pos, kind = ex._resolve_action_position("left", buf, 1.00, {})
    assert kind == "exact"
    np.testing.assert_array_equal(pos, [1.0, 1.0])
    assert ex.get_action_fill_stats().get("left", {}).get("stale", 0) == 0


def test_max_age_also_bounds_hold_last():
    ex = make_extractor(action_match="causal", action_max_age_s=0.05)
    ex._last_known_action["left"] = np.array([5.0, 5.0], dtype=np.float32)
    ex._last_known_action_ts["left"] = 0.50     # half a second old
    obs = {"left": {"pos": np.array([7.0, 7.0], dtype=np.float32)}}
    pos, kind = ex._resolve_action_position("left", deque(), 1.00, obs)
    assert kind == "fallback_to_observation"
    np.testing.assert_array_equal(pos, [7.0, 7.0])


def test_hold_last_is_unbounded_when_no_max_age_is_set():
    ex = make_extractor(action_match="causal")
    ex._last_known_action["left"] = np.array([5.0, 5.0], dtype=np.float32)
    ex._last_known_action_ts["left"] = 0.10     # ancient
    obs = {"left": {"pos": np.array([7.0, 7.0], dtype=np.float32)}}
    pos, kind = ex._resolve_action_position("left", deque(), 1.00, obs)
    assert kind == "hold_last"
    np.testing.assert_array_equal(pos, [5.0, 5.0])
