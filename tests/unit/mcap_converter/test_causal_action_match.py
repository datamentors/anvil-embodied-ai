"""Regression coverage for causal arm/gripper commands."""

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


def test_defaults_use_causal_commands():
    cfg = DataConfig()
    assert cfg.action_match == "causal"


def test_causal_never_reaches_forward():
    ex = make_extractor(action_match="causal")
    buf = _buffer([(0.990, [1.0, 1.0]), (1.001, [9.0, 9.0])])
    pos, kind = ex._resolve_action_position("left", buf, 1.000, {})
    assert kind == "exact"
    np.testing.assert_array_equal(pos, [1.0, 1.0])


def test_future_read_ahead_cannot_seed_causal_hold():
    ex = make_extractor(action_match="causal")
    ex._last_known_action["left"] = np.array([9.0, 9.0], dtype=np.float32)
    obs = {"left": {"pos": np.array([7.0, 7.0], dtype=np.float32)}}
    pos, kind = ex._resolve_action_position("left", _buffer([(2.25, [9.0, 9.0])]), 1.0, obs)
    assert kind == "fallback_to_observation"
    np.testing.assert_array_equal(pos, [7.0, 7.0])


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


@pytest.mark.parametrize("buffer_seconds", [0.5, 20.0])
def test_streaming_extraction_never_holds_a_future_command(buffer_seconds):
    from pathlib import Path

    fixtures = Path(__file__).resolve().parents[2] / "smoke" / "fixtures"
    namespace = DataConfig.__module__.split(".")[0]
    loader = importlib.import_module(f"{namespace}.config.loader").ConfigLoader
    config = loader.from_yaml(str(fixtures / "configs/mcap-converter-smoke-test-cmd.yaml"))
    config.action_match = "causal"
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
