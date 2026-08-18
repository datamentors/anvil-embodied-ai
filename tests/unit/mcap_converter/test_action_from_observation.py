"""End-to-end regression tests for action-from-observation alignment.

The AFO contract is ``action[t] = observation.state[t + N]``. These tests run
the real streaming extractor against the committed smoke MCAP, so buffering,
camera subsampling, command-topic selection, and episode flushing are covered
together.
"""

from collections import deque
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "smoke" / "fixtures"
MCAP_PATH = str(FIXTURES / "test-session" / "0001" / "0001_0.mcap")
AFO_CONFIG_PATH = str(FIXTURES / "configs" / "mcap-converter-smoke-test-afo.yaml")
CMD_CONFIG_PATH = str(FIXTURES / "configs" / "mcap-converter-smoke-test-cmd.yaml")

IMPLEMENTATIONS = [
    pytest.param("mcap_converter", id="cpu"),
    pytest.param("mcap_convert_gpu", id="gpu"),
]


def _load(
    namespace: str,
    config_path: str,
    *,
    fps: int,
    n: int | None = None,
    buffer_seconds: float = 5.0,
):
    config_loader = import_module(f"{namespace}.config.loader").ConfigLoader
    extractor_cls = import_module(f"{namespace}.core.extractor").BufferedStreamExtractor
    config = config_loader.from_yaml(config_path)
    config.image_resolution = [4, 4]
    if n is not None:
        config.action_from_observation_n = n
    return extractor_cls(
        config=config,
        fps=fps,
        buffer_seconds=buffer_seconds,
        quiet=True,
    )


def _make_bimanual_extractor(namespace: str):
    schema = import_module(f"{namespace}.config.schema")
    extractor_cls = import_module(f"{namespace}.core.extractor").BufferedStreamExtractor
    config = schema.DataConfig(
        action_topics={
            f"/commands/{arm}": schema.ActionTopicConfig(
                arm=arm,
                joint_order=["joint1", "joint2"],
            )
            for arm in ("left", "right")
        },
        action_from_observation=True,
        action_from_observation_n=2,
    )
    return extractor_cls(config=config, fps=1, quiet=True)


def _sample_buffer(offset: float):
    empty = np.array([], dtype=np.float32)
    return deque(
        (
            float(step),
            np.array([offset + step, offset + step + 0.25], dtype=np.float32),
            empty,
            empty,
        )
        for step in range(3)
    )


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("fps", "n", "buffer_seconds"),
    [
        (30, 10, 5.0),
        # half_buffer=150 exceeds the 120-frame fixture, forcing the entire
        # episode through the flush-only path while still subsampling to 15 fps.
        (15, 3, 20.0),
    ],
)
def test_afo_action_is_exactly_the_nth_future_output_observation(
    namespace: str,
    fps: int,
    n: int,
    buffer_seconds: float,
):
    baseline = list(_load(namespace, CMD_CONFIG_PATH, fps=fps).extract_frames(MCAP_PATH))
    reference = list(
        _load(
            namespace,
            CMD_CONFIG_PATH,
            fps=fps,
            buffer_seconds=buffer_seconds,
        ).extract_frames(MCAP_PATH)
    )
    actual = list(
        _load(
            namespace,
            AFO_CONFIG_PATH,
            fps=fps,
            n=n,
            buffer_seconds=buffer_seconds,
        ).extract_frames(MCAP_PATH)
    )

    assert len(baseline) > n
    assert len(reference) == len(baseline)
    assert len(actual) == len(baseline) - n

    for expected, flushed in zip(baseline, reference, strict=True):
        np.testing.assert_array_equal(
            flushed["observation.state"],
            expected["observation.state"],
        )
        np.testing.assert_array_equal(flushed["action"], expected["action"])

    for index, frame in enumerate(actual):
        np.testing.assert_array_equal(
            frame["observation.state"],
            baseline[index]["observation.state"],
        )
        np.testing.assert_array_equal(
            frame["action"],
            baseline[index + n]["observation.state"],
        )

    # The last valid action must use the episode's final observation. This
    # catches both off-by-one errors and silently clamping the final N frames.
    np.testing.assert_array_equal(
        actual[-1]["action"],
        baseline[-1]["observation.state"],
    )


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_afo_bimanual_order_and_command_buffer_isolation(namespace: str):
    extractor = _make_bimanual_extractor(namespace)
    joint_buffers = {
        ("observation", "left"): {"buffer": _sample_buffer(0.0)},
        ("observation", "right"): {"buffer": _sample_buffer(100.0)},
        # Even an already-populated command buffer must be ignored in AFO mode.
        ("action", "left"): {"buffer": _sample_buffer(-100.0)},
        ("action", "right"): {"buffer": _sample_buffer(-200.0)},
    }

    assert (
        extractor._align_joint_states(
            {("observation", "left"): {"buffer": _sample_buffer(0.0)}},
            target_ts=0.0,
        )
        is None
    )

    frames = [
        extractor._align_joint_states(joint_buffers, target_ts=float(step))
        for step in range(3)
    ]
    assert all(frame is not None and "action" not in frame for frame in frames)

    pending = deque()
    assert extractor._finalize_afo_frame(pending, frames[0]) is None
    assert extractor._finalize_afo_frame(pending, frames[1]) is None
    ready = extractor._finalize_afo_frame(pending, frames[2])

    assert ready is frames[0]
    np.testing.assert_array_equal(
        ready["observation.state"],
        np.array([0.0, 0.25, 100.0, 100.25], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        ready["action"],
        np.array([2.0, 2.25, 102.0, 102.25], dtype=np.float32),
    )


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
@pytest.mark.parametrize("n", [0, -1])
def test_afo_rejects_non_positive_lookahead(namespace: str, n: int):
    with pytest.raises(ValueError, match="action_from_observation_n must be positive"):
        _load(namespace, AFO_CONFIG_PATH, fps=30, n=n)


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_afo_rejects_frames_without_observation_state(namespace: str):
    extractor = _make_bimanual_extractor(namespace)
    error_cls = import_module(f"{namespace}.exceptions").DataExtractionError
    with pytest.raises(error_cls, match="observation.state"):
        extractor._finalize_afo_frame(deque(), {"task": "camera-only"})

    camera_buffers = {
        "main": deque([(0.0, np.zeros((2, 2, 3), dtype=np.uint8))]),
    }
    assert (
        extractor._align_frame_at_cursor(
            camera_buffers=camera_buffers,
            joint_buffers={},
            cursor=0,
            main_cam="main",
            task="camera-only",
            resize_func=lambda image, _size: image,
        )
        is None
    )


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_effective_afo_override_is_persisted(namespace: str, tmp_path: Path):
    config_loader = import_module(f"{namespace}.config.loader").ConfigLoader
    write_config = import_module(
        f"{namespace}.cli.convert"
    )._write_effective_conversion_config
    config = config_loader.from_yaml(AFO_CONFIG_PATH)
    config.action_from_observation_n = 3

    destination = tmp_path / "conversion_config.yaml"
    write_config(config, destination)
    restored = config_loader.from_yaml(str(destination))

    assert restored.action_from_observation is True
    assert restored.action_from_observation_n == 3
    assert restored.action_topics == config.action_topics
    assert restored.joint_name_pattern == config.joint_name_pattern
