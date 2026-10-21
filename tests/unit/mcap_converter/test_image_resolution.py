"""Tests for image resizing: the converter never letterbox-pads, crops or stretches.

The dataset width comes from ``image_resolution``; the height follows the
cameras' native aspect ratio. A camera with a different aspect ratio stops
the conversion instead of being padded.
"""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "smoke" / "fixtures"
MCAP_PATH = FIXTURES / "test-session" / "0001" / "0001_0.mcap"
CMD_CONFIG_PATH = str(FIXTURES / "configs" / "mcap-converter-smoke-test-cmd.yaml")

IMPLEMENTATIONS = [
    pytest.param("mcap_converter", id="cpu"),
    pytest.param("mcap_convert_gpu", id="gpu"),
]


def _image_utils(namespace: str):
    return import_module(f"{namespace}.utils.image_utils")


def _load_config(namespace: str):
    return import_module(f"{namespace}.config.loader").ConfigLoader.from_yaml(CMD_CONFIG_PATH)


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("source_size", "target_width", "expected"),
    [
        ((1920, 1080), 640, (640, 360)),
        ((1920, 1080), 480, (480, 270)),
        ((640, 480), 640, (640, 480)),
        ((848, 480), 640, (640, 362)),  # height rounded to even for yuv420p
    ],
)
def test_output_resolution_follows_camera_aspect_ratio(namespace, source_size, target_width, expected):
    assert _image_utils(namespace).output_resolution(source_size, target_width) == expected


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_resize_image_fills_frame_without_padding(namespace):
    image_utils = _image_utils(namespace)
    img = np.full((1080, 1920, 3), 200, dtype=np.uint8)

    resized = image_utils.resize_image(img, image_utils.output_resolution((1920, 1080), 640))

    assert resized.shape == (360, 640, 3)
    assert resized.min() == 200  # no black bars


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_resize_image_refuses_aspect_ratio_change(namespace):
    img = np.full((1080, 1920, 3), 200, dtype=np.uint8)

    # RuntimeError, not ValueError: the converter skips episodes on ValueError.
    with pytest.raises(RuntimeError, match="aspect ratio"):
        _image_utils(namespace).resize_image(img, (640, 480))


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_resolve_image_resolution_uses_camera_aspect_ratio(namespace):
    convert = import_module(f"{namespace}.cli.convert")
    config = _load_config(namespace)

    resolved = convert.resolve_image_resolution(config, [MCAP_PATH])

    # Smoke fixture cameras are 4×4: width 640 from the config, square height.
    assert resolved.image_resolution == [640, 640]
    assert config.image_resolution == [640, 480]


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_resolve_image_resolution_rejects_mixed_aspect_ratios(namespace, monkeypatch):
    convert = import_module(f"{namespace}.cli.convert")
    config = _load_config(namespace)
    monkeypatch.setattr(
        convert,
        "probe_camera_resolutions",
        lambda *_: {"waist": (1920, 1080), "wrist_r": (640, 480), "chest": (1920, 1080)},
    )

    with pytest.raises(ValueError, match="wrist_r 640x480"):
        convert.resolve_image_resolution(config, [MCAP_PATH])
