"""Tests for mcap-convert --undistort: lens undistortion with the session's calibration.

Frames are undistorted at native resolution with the intrinsics in
``<session>/meta/camera_calibration.json`` (alpha=0: no black borders), then
resized. The dataset gets a ``meta/camera_calibration.json`` with the
undistorted camera matrix and zero distortion. Without the flag nothing changes.
"""

import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "smoke" / "fixtures"
CMD_CONFIG_PATH = str(FIXTURES / "configs" / "mcap-converter-smoke-test-cmd.yaml")

IMPLEMENTATIONS = [
    pytest.param("mcap_converter", id="cpu"),
    pytest.param("mcap_convert_gpu", id="gpu"),
]

# Small synthetic camera with strong barrel distortion, so tests stay fast.
WIDTH, HEIGHT = 80, 60
CALIBRATION = {
    "cam_chest": {
        "camera_matrix": [[60.0, 0.0, 40.0], [0.0, 60.0, 30.0], [0.0, 0.0, 1.0]],
        "dist_coeffs": [-0.3, 0.1, 0.0, 0.0, 0.0],
        "width": WIDTH,
        "height": HEIGHT,
    }
}
TOPIC_MAPPING = {"/cam_chest/image_raw/compressed": "base"}


def _undistort(namespace: str):
    return import_module(f"{namespace}.core.undistort")


def _session(tmp_path: Path, calibration: dict = CALIBRATION) -> Path:
    (tmp_path / "meta").mkdir(parents=True)
    (tmp_path / "meta" / "camera_calibration.json").write_text(json.dumps(calibration))
    return tmp_path


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_source_camera_name_is_first_topic_segment(namespace):
    assert _undistort(namespace).source_camera_name("/cam_chest/image_raw/compressed") == "cam_chest"
    assert _undistort(namespace).source_camera_name("/cam_wrist_l/image_raw") == "cam_wrist_l"


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_build_undistortion_requires_session_calibration(namespace, tmp_path):
    with pytest.raises(FileNotFoundError, match="camera_calibration.json"):
        _undistort(namespace).build_undistortion(str(tmp_path), TOPIC_MAPPING)


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_build_undistortion_rejects_missing_camera(namespace, tmp_path):
    mapping = {**TOPIC_MAPPING, "/cam_wrist_l/image_raw/compressed": "left_wrist"}

    with pytest.raises(ValueError, match="cam_wrist_l"):
        _undistort(namespace).build_undistortion(str(_session(tmp_path)), mapping)


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_undistort_image_keeps_size_without_black_borders(namespace, tmp_path):
    undistort = _undistort(namespace)
    cameras = undistort.build_undistortion(str(_session(tmp_path)), TOPIC_MAPPING)
    img = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)

    out = undistort.undistort_image(img, cameras["base"]["maps"])

    assert out.shape == img.shape
    assert out.min() > 150  # alpha=0: no pixel is sampled from outside the frame


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_undistort_image_refuses_other_resolution(namespace, tmp_path):
    undistort = _undistort(namespace)
    cameras = undistort.build_undistortion(str(_session(tmp_path)), TOPIC_MAPPING)
    img = np.zeros((HEIGHT * 2, WIDTH * 2, 3), dtype=np.uint8)

    # RuntimeError, not ValueError: the converter skips episodes on ValueError.
    with pytest.raises(RuntimeError, match="calibration"):
        undistort.undistort_image(img, cameras["base"]["maps"])


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_write_output_calibration_stores_undistorted_matrix(namespace, tmp_path):
    undistort = _undistort(namespace)
    cameras = undistort.build_undistortion(str(_session(tmp_path / "raw")), TOPIC_MAPPING)
    output_dir = tmp_path / "dataset"

    path = undistort.write_output_calibration(str(output_dir), cameras, [40, 30])

    assert path == output_dir / "meta" / "camera_calibration.json"
    entry = json.loads(path.read_text())["observation.images.base"]
    assert entry["source"] == "cam_chest"
    assert entry["undistorted"] is True
    assert entry["alpha"] == 0.0
    np.testing.assert_allclose(entry["camera_matrix"], cameras["base"]["K_new"])
    assert entry["dist_coeffs"] == [0.0] * 5
    assert (entry["width"], entry["height"]) == (WIDTH, HEIGHT)
    assert (entry["output_width"], entry["output_height"]) == (40, 30)


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_undistort_is_off_by_default(namespace):
    convert = import_module(f"{namespace}.cli.convert")

    assert convert.build_parser().parse_args(["-i", "session"]).undistort is False
    assert convert.build_parser().parse_args(["-i", "session", "--undistort"]).undistort is True


@pytest.mark.parametrize("namespace", IMPLEMENTATIONS)
def test_extractor_without_maps_only_resizes(namespace):
    extractor_module = import_module(f"{namespace}.core.extractor")
    config = import_module(f"{namespace}.config.loader").ConfigLoader.from_yaml(CMD_CONFIG_PATH)
    extractor = extractor_module.BufferedStreamExtractor(config)
    calls = []

    def resize(img, size):
        calls.append(size)
        return img

    img = np.full((HEIGHT, WIDTH, 3), 7, dtype=np.uint8)

    assert extractor._prepare_image("base", img, resize) is img
    assert calls == [extractor.target_size]
