"""Frame sources: one per robot I/O stack, picked by the config's ``source.type``.

Sources are read-only: getters and subscriptions only. None of them publishes,
commands motion or changes a camera or control setting.

To support a robot that is neither GDK nor rosbridge, write a class that
implements :class:`Source`, then register it in :func:`make_source` and in
``config.SOURCES``.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .common import K_D_from_intrinsic, T_to_pose7, make_T, pose_delta, quat_to_mat, write_json

log = logging.getLogger("camera_calib.source")


@dataclass
class Frame:
    """One camera image.

    Attributes:
        gray: Decoded grayscale image, used for detection.
        jpg: The original JPEG bytes, saved unchanged as ``<cam>.jpg``.
        stamp_ns: Capture timestamp, when the source provides one.
    """

    gray: np.ndarray
    jpg: bytes
    stamp_ns: int | None = None


@dataclass
class Tick:
    """One read of every camera plus the robot's motion state.

    Attributes:
        frames: Camera name -> frame, or None when that camera had no usable image.
        motion: Source-specific snapshot; only the same source's ``still()`` reads it.
    """

    frames: dict[str, Frame | None]
    motion: Any


class Source(Protocol):
    """What the capture loop needs from a source."""

    last_error: str  # why the last read() returned None

    def write_factory(self, run_dir: Path) -> dict:
        """Write ``factory.json`` (factory intrinsics, or a note) and return it."""

    def run_info(self) -> dict:
        """Return extra fields for ``run.json``."""

    def read(self) -> Tick | None:
        """Return one tick, or None when there is no fresh data."""

    def still(self, prev: Any, cur: Any) -> bool:
        """Return True if the robot did not move between two motion snapshots."""

    def K_D(self, cam: str, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray | None]:
        """Return a camera model for the capture-time "new view" check (never a result)."""

    def sample_meta(self, tick: Tick, cams: list[str]) -> dict:
        """Return what ``meta.json`` records next to the saved images."""

    def close(self) -> None:
        """Release the connection."""


def make_source(cfg: dict, cams: list[str]) -> Source:
    """Create the source named by ``cfg["source"]["type"]``.

    Args:
        cfg: Validated robot config.
        cams: Cameras to read.

    Returns:
        A connected source.
    """
    if cfg["source"]["type"] == "gdk":
        return GdkSource(cfg, cams)
    return RosbridgeSource(cfg, cams)


def nominal_K(w: int, h: int) -> np.ndarray:
    """Rough ~90 degree HFOV pinhole, only for judging "new view" when no factory K exists."""
    f = w / 2.0
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)


# -- AgiBot GDK ---------------------------------------------------------------------
def _gdk_T(tr: Any) -> np.ndarray:
    r, t = tr.rotation, tr.translation
    return make_T(quat_to_mat(r.x, r.y, r.z, r.w), [t.x, t.y, t.z])


def _gdk_frame(img: Any) -> Frame | None:
    """GDK image -> Frame, or None if it is empty or not JPEG."""
    data = getattr(img, "data", None)
    if data is None or len(data) == 0:
        return None
    if "JPEG" not in str(img.encoding):
        log.warning("unsupported encoding %s", img.encoding)
        return None
    jpg = data.tobytes() if hasattr(data, "tobytes") else bytes(data)
    gray = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
    return None if gray is None else Frame(gray, jpg, int(img.timestamp_ns))


class GdkSource:
    """Images, TF and factory intrinsics through AgiBot's GDK getters.

    Args:
        cfg: Robot config with ``source.type: gdk``.
        cams: Cameras to read.
    """

    def __init__(self, cfg: dict, cams: list[str]) -> None:
        from .gdk_env import import_gdk

        self.cfg, self.cams = cfg, cams
        self.table = cfg["cameras"]
        # Stillness watches every camera's parent frame, not only the selected ones.
        self.tf_frames = sorted({c["parent"] for c in self.table.values()})
        still = cfg["stillness"]
        self.check_still = still["type"] == "tf"
        self.still_mm = float(still.get("still_mm", 0.0))
        self.still_deg = float(still.get("still_deg", 0.0))
        self.timeout_ms = float(cfg["source"].get("image_timeout_ms", 200.0))
        self.last_error = ""
        self._KD: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.g = import_gdk()
        log.info("gdk_init -> %s", self.g.gdk_init())
        try:
            self.cam = self.g.Camera()
            self.tf = self.g.TF()
            self.types = {c: getattr(self.g.CameraType, self.table[c]["gdk"]) for c in cams}
        except Exception:
            self.close()
            raise
        time.sleep(3.0)  # DDS needs a moment before the first sample arrives

    def close(self) -> None:
        """Release the GDK."""
        try:
            self.g.gdk_release()
        except Exception:  # noqa: BLE001
            pass

    def write_factory(self, run_dir: Path) -> dict:
        """Record the live factory intrinsics and the vendor files they come from.

        Args:
            run_dir: Run folder.

        Returns:
            The ``factory.json`` contents.
        """
        param_dir = Path(self.cfg["source"].get("vendor_param_dir", "/data/parameters/sensor"))
        factory: dict = {"cameras": {}, "vendor_files": {}}
        for name, c in self.table.items():
            entry: dict = {}
            try:
                ci = self.cam.get_camera_intrinsic(getattr(self.g.CameraType, c["gdk"]))
                entry["intrinsic"] = [float(v) for v in ci.intrinsic]
                entry["distortion"] = [float(v) for v in ci.distortion]
            except Exception as exc:  # noqa: BLE001 - the SDK throws for unsupported cameras
                entry["intrinsic_error"] = str(exc)
            factory["cameras"][name] = entry
            fname = c.get("vendor_intrinsic")
            if fname:
                try:
                    factory["vendor_files"][fname] = (param_dir / fname).read_text()
                except OSError as exc:
                    factory["vendor_files"][fname] = f"ERROR: {exc}"
            if name in self.cams and "intrinsic" in entry:
                self._KD[name] = K_D_from_intrinsic(entry["intrinsic"], entry["distortion"])
        write_json(run_dir / "factory.json", factory)
        log.info("factory calibration -> %s", run_dir / "factory.json")
        return factory

    def run_info(self) -> dict:
        """Check every selected camera has factory intrinsics (needed for the new-view check).

        Raises:
            SystemExit: Naming the cameras without factory intrinsics.
        """
        missing = [c for c in self.cams if c not in self._KD]
        if missing:
            raise SystemExit(f"no factory intrinsics for {missing} (see factory.json)")
        return {}

    def _read_tf(self) -> dict[str, np.ndarray]:
        return {f: _gdk_T(self.tf.get_tf_from_base_link(f)) for f in self.tf_frames}

    def read(self) -> Tick | None:
        """Read TF, every camera, then TF again; motion is the TF pair."""
        try:
            T1 = self._read_tf()
            imgs = {c: self.cam.get_latest_image(self.types[c], self.timeout_ms) for c in self.cams}
            T2 = self._read_tf()
        except Exception as exc:  # noqa: BLE001 - the SDK throws on transient DDS gaps
            self.last_error = f"read failed: {exc}"
            return None
        frames = {c: (_gdk_frame(img) if img is not None else None) for c, img in imgs.items()}
        return Tick(frames, (T1, T2))

    def _within(self, A: np.ndarray, B: np.ndarray) -> bool:
        dt, dr = pose_delta(A, B)
        return dt * 1000.0 <= self.still_mm and dr <= self.still_deg

    def still(self, prev: Any, cur: Any) -> bool:
        """Still across two ticks: previous end -> this start -> this end, for every frame.

        Always True with ``stillness.type: none``.
        """
        if not self.check_still:
            return True
        if prev is None:
            return False
        T1, T2 = cur
        return all(
            self._within(prev[1][f], T1[f]) and self._within(T1[f], T2[f]) for f in self.tf_frames
        )

    def K_D(self, cam: str, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray | None]:
        """Factory camera model."""
        return self._KD[cam]

    def sample_meta(self, tick: Tick, cams: list[str]) -> dict:
        """TF at the end of the tick and the saved images' timestamps."""
        return {
            "tf": {f: T_to_pose7(T) for f, T in tick.motion[1].items()},
            "image_ts_ns": {c: tick.frames[c].stamp_ns for c in cams},
        }


# -- ROS 2 over rosbridge -------------------------------------------------------------
class RosbridgeFeed:
    """Keep the latest JPEG per camera topic and the latest joint positions.

    A background thread subscribes over the rosbridge websocket and reconnects
    on any error.

    Args:
        url: rosbridge websocket URL.
        topics: Camera name -> topic publishing ``sensor_msgs/msg/CompressedImage``.
        joints_topic: Topic publishing ``sensor_msgs/msg/JointState``, or None to skip.
        throttle_ms: Minimum interval between frames per camera.
    """

    def __init__(
        self, url: str, topics: dict[str, str], joints_topic: str | None, throttle_ms: int
    ) -> None:
        self.url = url
        self.topics = topics
        self.joints_topic = joints_topic
        self.throttle_ms = throttle_ms
        self._cam_of = {t: c for c, t in topics.items()}
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[bytes, float]] = {}
        self._joints: tuple[np.ndarray, float] | None = None
        self.connected = False
        self.last_error = ""
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def close(self) -> None:
        """Stop the background thread."""
        self._stop = True

    def frame(self, cam: str, max_age_s: float) -> bytes | None:
        """Latest JPEG for a camera, or None if none arrived within ``max_age_s``."""
        with self._lock:
            item = self._frames.get(cam)
        if item is None or time.monotonic() - item[1] > max_age_s:
            return None
        return item[0]

    def joints(self, max_age_s: float) -> np.ndarray | None:
        """Latest joint positions, or None if none arrived within ``max_age_s``."""
        with self._lock:
            item = self._joints
        if item is None or time.monotonic() - item[1] > max_age_s:
            return None
        return item[0]

    def _run(self) -> None:
        from websockets.sync.client import connect

        backoff = 1.0
        while not self._stop:
            try:
                with connect(self.url, max_size=2**26, open_timeout=10) as ws:
                    for topic in self.topics.values():
                        self._subscribe(
                            ws, topic, "sensor_msgs/msg/CompressedImage", self.throttle_ms
                        )
                    if self.joints_topic is not None:
                        self._subscribe(ws, self.joints_topic, "sensor_msgs/msg/JointState", 100)
                    self.connected, backoff = True, 1.0
                    log.info("rosbridge connected: %s", self.url)
                    while not self._stop:
                        self._handle(json.loads(ws.recv(timeout=5)))
            except Exception as exc:  # noqa: BLE001 - reconnect on anything
                self.connected = False
                self.last_error = str(exc)
                log.warning("rosbridge: %s (retry in %.0f s)", exc, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

    @staticmethod
    def _subscribe(ws: Any, topic: str, msg_type: str, throttle_ms: int) -> None:
        ws.send(
            json.dumps(
                {"op": "subscribe", "topic": topic, "type": msg_type,
                 "throttle_rate": throttle_ms, "queue_length": 1}
            )
        )  # fmt: skip

    def _handle(self, m: dict) -> None:
        if m.get("op") != "publish":
            return
        topic, msg, now = m.get("topic", ""), m.get("msg", {}), time.monotonic()
        if topic is not None and topic == self.joints_topic:
            pos = np.asarray(msg.get("position", []), dtype=np.float64)
            if pos.size:
                with self._lock:
                    self._joints = (pos, now)
            return
        cam, data = self._cam_of.get(topic), msg.get("data")
        if cam is not None and isinstance(data, str):
            with self._lock:
                self._frames[cam] = (base64.b64decode(data), now)


V4L2_CTRLS = "focus_automatic_continuous,focus_absolute,auto_exposure,exposure_time_absolute"


def camera_state(devices: dict[str, str]) -> dict:
    """Read v4l2 focus and exposure per camera; intrinsics are only valid at this focus.

    Args:
        devices: Camera name -> v4l2 device path.

    Returns:
        Camera name -> control values (or ``{"error": ...}``).
    """
    out: dict = {}
    for cam, dev in devices.items():
        try:
            res = subprocess.run(
                ["v4l2-ctl", "-d", dev, f"--get-ctrl={V4L2_CTRLS}"],
                capture_output=True, text=True, timeout=5, check=False,
            )  # fmt: skip
            out[cam] = dict(
                line.split(": ", 1) for line in res.stdout.strip().splitlines() if ": " in line
            )
            if res.returncode != 0:
                out[cam]["error"] = res.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            out[cam] = {"error": str(exc)}
    return out


class RosbridgeSource:
    """Camera and joint-state topics over rosbridge; topic names come from the config.

    Cameras must publish ``sensor_msgs/msg/CompressedImage`` (JPEG); the stillness
    topic, if used, ``sensor_msgs/msg/JointState``.

    Args:
        cfg: Robot config with ``source.type: rosbridge``.
        cams: Cameras to read.
    """

    def __init__(self, cfg: dict, cams: list[str]) -> None:
        self.cfg, self.cams = cfg, cams
        src, still = cfg["source"], cfg["stillness"]
        self.check_still = still["type"] == "joint_states"
        self.still_rad = float(still.get("still_rad", 0.0))
        topics = {c: cfg["cameras"][c]["topic"] for c in cams}
        joints_topic = still["topic"] if self.check_still else None
        self.feed = RosbridgeFeed(
            src["url"], topics, joints_topic, int(src.get("throttle_ms", 200))
        )
        self.last_error = ""

    def close(self) -> None:
        """Stop the feed."""
        self.feed.close()

    def write_factory(self, run_dir: Path) -> dict:
        """Record that this robot has no factory intrinsics."""
        note = f"{self.cfg['robot']}: no factory intrinsics"
        if self.cfg.get("factory_note"):
            note += f" ({self.cfg['factory_note']})"
        factory = {"cameras": {}, "note": note}
        write_json(run_dir / "factory.json", factory)
        return factory

    def run_info(self) -> dict:
        """Record each camera's v4l2 focus/exposure; warn if autofocus is on."""
        devices = {
            c: self.cfg["cameras"][c]["v4l2_device"]
            for c in self.cams
            if self.cfg["cameras"][c].get("v4l2_device")
        }
        state = camera_state(devices)
        for cam, s in state.items():
            if s.get("focus_automatic_continuous") not in (None, "0"):
                log.warning(
                    "%s: AUTOFOCUS IS ON -- intrinsics change with focus; results invalid", cam
                )
        return {"camera_state": state}

    def read(self) -> Tick | None:
        """Latest frame per camera and latest joints (each at most 1 s old)."""
        q = self.feed.joints(max_age_s=1.0) if self.check_still else None
        jpgs = {c: self.feed.frame(c, max_age_s=1.0) for c in self.cams}
        if (self.check_still and q is None) or all(j is None for j in jpgs.values()):
            feed = self.feed
            self.last_error = (
                f"no fresh data from rosbridge (connected={feed.connected} {feed.last_error})"
            )
            return None
        frames: dict[str, Frame | None] = {}
        for c, j in jpgs.items():
            gray = (
                None
                if j is None
                else cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_GRAYSCALE)
            )
            frames[c] = None if gray is None else Frame(gray, j)
        return Tick(frames, q)

    def still(self, prev: Any, cur: Any) -> bool:
        """No joint moved more than ``still_rad`` since the previous tick.

        Always True with ``stillness.type: none``.
        """
        if not self.check_still:
            return True
        return (
            prev is not None
            and prev.shape == cur.shape
            and float(np.max(np.abs(cur - prev))) <= self.still_rad
        )

    def K_D(self, cam: str, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray | None]:
        """Nominal pinhole; this robot has no factory intrinsics."""
        return nominal_K(shape[1], shape[0]), None

    def sample_meta(self, tick: Tick, cams: list[str]) -> dict:
        """Joint positions at save time (empty without joint stillness)."""
        joints = [] if tick.motion is None else [float(v) for v in tick.motion]
        return {"tf": {}, "joints": joints}
