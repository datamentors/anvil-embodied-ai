"""Causal selection, held targets, and explicit nearest compatibility."""

import importlib
import sys
from collections import deque

import numpy as np
import pytest

from mcap_converter.config.schema import ActionTopicConfig, DataConfig
from mcap_converter.core.extractor import BufferedStreamExtractor


@pytest.fixture(autouse=True, params=["mcap_converter", "mcap_convert_gpu"])
def converter_package(request, monkeypatch):
    """Both deployed extractor implementations must preserve causality."""
    schema = importlib.import_module(f"{request.param}.config.schema")
    extractor = importlib.import_module(f"{request.param}.core.extractor")
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "DataConfig", schema.DataConfig)
    monkeypatch.setattr(module, "ActionTopicConfig", schema.ActionTopicConfig)
    monkeypatch.setattr(module, "BufferedStreamExtractor", extractor.BufferedStreamExtractor)


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


def test_defaults_use_causal_commands():
    cfg = DataConfig()
    assert cfg.action_match == "causal"
    assert cfg.action_max_age_s is None


def test_default_nearest_can_select_a_command_from_the_future():
    """The behaviour causal mode exists to avoid, pinned so it stays visible."""
    ex = make_extractor(action_match="nearest")
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
    ex._remember_causal_action("left", 0.5, np.array([5.0, 5.0], dtype=np.float32))
    buf = _buffer([(1.10, [9.0, 9.0])])
    pos, kind = ex._resolve_action_position("left", buf, 1.00, {})
    assert kind == "hold_last"
    np.testing.assert_array_equal(pos, [5.0, 5.0])


# --------------------------------------------------------------------------- #
# maximum age


def test_stale_buffer_command_is_held_even_without_a_cached_command():
    ex = make_extractor(action_match="causal", action_max_age_s=0.05)
    buf = _buffer([(0.80, [1.0, 1.0])])  # 200 ms old
    obs = {"left": {"pos": np.array([7.0, 7.0], dtype=np.float32)}}
    pos, kind = ex._resolve_action_position("left", buf, 1.00, obs)
    assert kind == "hold_last"
    np.testing.assert_array_equal(pos, [1.0, 1.0])
    assert ex.get_action_fill_stats()["left"]["stale"] == 1


def test_max_age_accepts_a_command_inside_the_window():
    ex = make_extractor(action_match="causal", action_max_age_s=0.05)
    buf = _buffer([(0.97, [1.0, 1.0])])  # 30 ms old
    pos, kind = ex._resolve_action_position("left", buf, 1.00, {})
    assert kind == "exact"
    np.testing.assert_array_equal(pos, [1.0, 1.0])
    assert ex.get_action_fill_stats().get("left", {}).get("stale", 0) == 0


def test_hold_last_is_never_bounded_by_max_age():
    """A parked arm is holding its last command, so that command is the action.

    Substituting the measured pose here would swap a correct value for the
    commanded pose plus gravity sag — wrong in the same direction on every
    parked frame. Measured on these recordings: an uncommanded arm moves
    0.08-0.23 rad against 1.3-1.5 rad while commanded.
    """
    ex = make_extractor(action_match="causal", action_max_age_s=0.05)
    ex._remember_causal_action("left", 0.5, np.array([5.0, 5.0], dtype=np.float32))
    obs = {"left": {"pos": np.array([7.0, 7.0], dtype=np.float32)}}
    pos, kind = ex._resolve_action_position("left", deque(), 1.00, obs)
    assert kind == "hold_last"
    np.testing.assert_array_equal(pos, [5.0, 5.0])


def test_a_stale_command_comes_back_through_hold_last_not_as_observation():
    """The realistic shape: a command was buffered, then the arm was released.

    max_age stops it being reported as "exact" and tallies it as stale, but the
    value returned is that same command — not the measured joint position.
    """
    ex = make_extractor(action_match="causal", action_max_age_s=0.05)
    ex._remember_causal_action("left", 0.8, np.array([1.0, 1.0], dtype=np.float32))
    buf = _buffer([(0.80, [1.0, 1.0])])  # 200 ms old
    obs = {"left": {"pos": np.array([7.0, 7.0], dtype=np.float32)}}
    pos, kind = ex._resolve_action_position("left", buf, 1.00, obs)
    assert kind == "hold_last"
    np.testing.assert_array_equal(pos, [1.0, 1.0])
    assert ex.get_action_fill_stats()["left"]["stale"] == 1


def test_observation_fallback_only_when_the_arm_never_commanded():
    """The one case with no command to hold: the head of an episode."""
    ex = make_extractor(action_match="causal", action_max_age_s=0.05)
    obs = {"left": {"pos": np.array([7.0, 7.0], dtype=np.float32)}}
    pos, kind = ex._resolve_action_position("left", deque(), 1.00, obs)
    assert kind == "fallback_to_observation"
    np.testing.assert_array_equal(pos, [7.0, 7.0])


def test_future_read_ahead_cannot_seed_causal_hold():
    ex = make_extractor(action_match="causal")
    ex._last_known_action["left"] = np.array([9.0, 9.0], dtype=np.float32)
    obs = {"left": {"pos": np.array([7.0, 7.0], dtype=np.float32)}}
    pos, kind = ex._resolve_action_position("left", _buffer([(2.25, [9.0, 9.0])]), 1.0, obs)
    assert kind == "fallback_to_observation"
    np.testing.assert_array_equal(pos, [7.0, 7.0])


def test_stale_past_command_is_held_instead_of_future_read_ahead():
    ex = make_extractor(action_match="causal", action_max_age_s=0.05)
    ex._last_known_action["left"] = np.array([9.0, 9.0], dtype=np.float32)
    pos, kind = ex._resolve_action_position(
        "left", _buffer([(0.8, [1.0, 2.0]), (2.25, [9.0, 9.0])]), 1.0, {}
    )
    assert kind == "hold_last"
    np.testing.assert_array_equal(pos, [1.0, 2.0])


def test_eviction_keeps_past_command_and_gripper_together():
    ex = make_extractor(action_match="causal")
    old = [0.01, 1, 2, 3, 4, 5, 6, 7]
    future = [0.05, 9, 9, 9, 9, 9, 9, 9]
    buffers = {("action", "left"): {"buffer": _buffer([(0.1, old), (2.25, future)])}}
    ex._last_known_action["left"] = np.array(future, dtype=np.float32)
    ex._sync_joint_buffers(buffers, 0.5)
    pos, kind = ex._resolve_action_position("left", buffers[("action", "left")]["buffer"], 1.0, {})
    assert kind == "hold_last"
    np.testing.assert_array_equal(pos, np.array(old, dtype=np.float32))


def test_future_held_cache_is_rejected_for_an_earlier_frame():
    ex = make_extractor(action_match="causal")
    ex._remember_causal_action("left", 2.0, np.array([9.0, 9.0], dtype=np.float32))
    obs = {"left": {"pos": np.array([7.0, 7.0], dtype=np.float32)}}
    pos, kind = ex._resolve_action_position("left", deque(), 1.0, obs)
    assert kind == "fallback_to_observation"
    np.testing.assert_array_equal(pos, [7.0, 7.0])


@pytest.mark.parametrize("buffer_seconds", [0.5, 20.0])
def test_streaming_extraction_never_holds_a_future_command(buffer_seconds):
    from pathlib import Path

    fixtures = Path(__file__).resolve().parents[2] / "smoke" / "fixtures"
    namespace = DataConfig.__module__.split(".")[0]
    loader = importlib.import_module(f"{namespace}.config.loader").ConfigLoader
    config = loader.from_yaml(str(fixtures / "configs/mcap-converter-smoke-test-cmd.yaml"))
    config.action_match = "causal"
    config.action_max_age_s = 0.05
    config.image_resolution = [4, 4]
    selected = []

    class TracedExtractor(BufferedStreamExtractor):
        def _resolve_action_position(self, robot, buffer, target_ts, obs_data):
            pos, kind = super()._resolve_action_position(robot, buffer, target_ts, obs_data)
            if kind in {"exact", "hold_last"}:
                timestamp, _ = self._last_causal_action[robot]
                selected.append(timestamp)
                assert timestamp <= target_ts
            return pos, kind

    extractor = TracedExtractor(config=config, fps=30, buffer_seconds=buffer_seconds, quiet=True)
    frames = list(extractor.extract_frames(str(fixtures / "test-session/0001/0001_0.mcap")))
    assert frames
    assert selected


def test_normal_configs_use_commands_and_afo_configs_remain_opt_in():
    from pathlib import Path

    namespace = DataConfig.__module__.split(".")[0]
    loader = importlib.import_module(f"{namespace}.config.loader").ConfigLoader
    configs = Path(__file__).resolve().parents[3] / "configs/mcap_converter"
    for path in configs.glob("openarm*quest*.yaml"):
        cfg = loader.from_yaml(str(path))
        assert cfg.action_from_observation == ("_afo" in path.name)
        assert cfg.action_match == "causal"


def test_eviction_cannot_replace_a_newer_causal_target_with_an_older_one():
    ex = make_extractor(action_match="causal")
    latest = np.array([2.0, 3.0], dtype=np.float32)
    ex._remember_causal_action("left", 0.9, latest)
    latest[:] = 99
    buffers = {("action", "left"): {"buffer": _buffer([(0.1, [1.0, 1.0])])}}
    ex._sync_joint_buffers(buffers, 0.5)
    pos, kind = ex._resolve_action_position("left", deque(), 1.0, {})
    assert kind == "hold_last"
    np.testing.assert_array_equal(pos, [2.0, 3.0])
