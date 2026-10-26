"""
Lens undistortion of camera frames (mcap-convert, on by default; --no-undistort skips it).

Reads the session's intrinsic calibration from <session>/meta/camera_calibration.json
(pinhole K + plumb_bob D per raw camera, e.g. cam_chest, at native resolution),
builds the remap tables once per camera, and undistorts each frame at native
resolution before it is resized. The dataset gets its own
meta/camera_calibration.json with the undistorted camera matrix (K_new, at
native resolution) and zero distortion.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

# Calibration attached to a raw session at upload. Keys are raw camera names.
SESSION_CALIBRATION = Path("meta") / "camera_calibration.json"

# Free scaling parameter of cv2.getOptimalNewCameraMatrix. 0 keeps only valid
# pixels: no black borders, at the cost of the outermost field of view.
ALPHA = 0.0


def source_camera_name(topic: str) -> str:
    """Raw camera name of an image topic: '/cam_chest/image_raw/compressed' -> 'cam_chest'."""
    return topic.strip("/").split("/")[0]


def build_undistortion(input_dir: str, camera_topic_mapping: Dict[str, str]) -> Dict[str, dict]:
    """
    Undistortion of every dataset camera, computed once per conversion.

    Args:
        input_dir: Raw session directory (holds meta/camera_calibration.json)
        camera_topic_mapping: Config mapping of image topic to dataset camera name

    Returns:
        {camera_name: {"source", "D", "size", "K_new", "maps"}}, where size is
        the calibrated (width, height), K_new the matrix of the undistorted
        native-resolution frame and maps the cv2.remap tables.
    """
    path = Path(input_dir) / SESSION_CALIBRATION
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist: undistortion needs the session's camera calibration. "
            "Pass --no-undistort to convert without undistorting."
        )
    with open(path) as f:
        calibration = json.load(f)

    cameras = {}
    for topic, cam_name in camera_topic_mapping.items():
        source = source_camera_name(topic)
        if source not in calibration:
            raise ValueError(
                f"{path} has no calibration for {source} (camera {cam_name}); "
                f"it has: {', '.join(sorted(calibration))}"
            )
        entry = calibration[source]
        K = np.array(entry["camera_matrix"], dtype=np.float64)
        D = np.array(entry["dist_coeffs"], dtype=np.float64)
        size = (int(entry["width"]), int(entry["height"]))
        K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, size, ALPHA, size)
        maps = cv2.initUndistortRectifyMap(K, D, None, K_new, size, cv2.CV_16SC2)
        cameras[cam_name] = {"source": source, "D": D, "size": size, "K_new": K_new, "maps": maps}
    return cameras


def undistort_image(img: np.ndarray, maps: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Undistort one native-resolution frame with tables from build_undistortion.

    Raises:
        RuntimeError: If the frame is not at the calibrated resolution. Not a
            ValueError: the converter skips episodes on ValueError (corrupt frames),
            whereas a calibration/camera mismatch must stop the run.
    """
    map1, map2 = maps
    if img.shape[:2] != map1.shape[:2]:
        raise RuntimeError(
            f"Frame is {img.shape[1]}x{img.shape[0]} but the camera calibration is for "
            f"{map1.shape[1]}x{map1.shape[0]}"
        )
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)


def write_output_calibration(output_dir: str, cameras: Dict[str, dict], output_size: tuple) -> Path:
    """
    Write <dataset>/meta/camera_calibration.json for undistorted frames.

    camera_matrix is K_new at the native resolution (width/height); it is not
    scaled to the stored frames (output_width/output_height). dist_coeffs are
    zero, because the stored frames are already undistorted. The source
    calibration stays in the raw session's meta/camera_calibration.json.
    """
    record = {}
    for cam_name, cam in cameras.items():
        record[f"observation.images.{cam_name}"] = {
            "source": cam["source"],
            "undistorted": True,
            "alpha": ALPHA,
            "camera_matrix": cam["K_new"].tolist(),
            "dist_coeffs": [0.0] * len(cam["D"]),
            "width": cam["size"][0],
            "height": cam["size"][1],
            "output_width": int(output_size[0]),
            "output_height": int(output_size[1]),
        }

    path = Path(output_dir) / "meta" / "camera_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path
