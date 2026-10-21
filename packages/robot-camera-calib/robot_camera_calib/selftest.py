"""Synthetic self-test: known ground truth through the real code. No robot needed.

Usage::

    python3 -m robot_camera_calib.selftest
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .capture import run_loop
from .common import Board, make_T, mat_to_quat, quat_to_mat
from .config import ROBOTS_DIR, load_robot, select_cams, validate
from .solve import View, camera_info_yaml, project, solve_intrinsics
from .sources import Frame, Tick, nominal_K

RNG = np.random.default_rng(0)

# G2 factory values (hand_left, head): realistic scale.
K_HAND = np.array([[479.70, 0, 640.50], [0, 479.46, 524.10], [0, 0, 1]])
D_HAND = np.array([-0.0478, -0.0103, -9.4e-05, 7.0e-05, 0.00196])
K_HEAD = np.array([[305.02, 0, 316.53], [0, 305.01, 203.46], [0, 0, 1]])
D_HEAD = np.array([-0.0306, 0.0336, 0.00014, -5.6e-05, -0.0119])

BOARD = Board("charuco:7x5:0.035:0.026:4x4_50")
CORNERS = np.array(
    [[x * 0.035, y * 0.035, 0.0] for y in range(1, 5) for x in range(1, 7)], np.float32
)
M_TRUE = make_T(np.diag([1.0, -1.0, -1.0]), [0.55, -0.10, 0.75])  # board pose in the world

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    """Print one PASS/FAIL line and remember failures."""
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILS.append(name)


# -- synthetic views ----------------------------------------------------------------
def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera pose (optical frame: z forward, y down) looking at ``target``."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, -1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    return make_T(np.column_stack([x, np.cross(z, x), z]), eye)


def _random_cam_poses(n: int, dist: tuple[float, float], jitter: float) -> list[np.ndarray]:
    centre = M_TRUE[:3, 3] + M_TRUE[:3, :3] @ np.array([0.1225, 0.0875, 0.0])
    poses = []
    for _ in range(n):
        d = RNG.uniform(*dist)
        th, ph = RNG.uniform(0, 2 * np.pi), RNG.uniform(np.radians(5), np.radians(40))
        eye = centre + d * np.array([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)])
        C = _look_at(eye, centre + RNG.normal(0, jitter, 3))
        roll = make_T(cv2.Rodrigues(np.array([0, 0, RNG.uniform(-0.6, 0.6)]))[0], [0, 0, 0])
        poses.append(C @ roll)
    return poses


def _synth_views(
    K: np.ndarray, D: np.ndarray, poses: list[np.ndarray], noise_px: float, size: tuple[int, int]
) -> list[View]:
    views = []
    for i, C in enumerate(poses):
        img = project(CORNERS, np.linalg.inv(C) @ M_TRUE, K, D)
        inside = (img[:, 0] > 0) & (img[:, 0] < size[0]) & (img[:, 1] > 0) & (img[:, 1] < size[1])
        if inside.sum() >= 6:
            img = (img + RNG.normal(0, noise_px, img.shape)).astype(np.float32)
            views.append(View(f"{i:04d}", CORNERS[inside], img[inside]))
    return views


def _wrong_seed(K: np.ndarray) -> np.ndarray:
    """A "factory" K that is off, so recovery cannot come from the seed itself."""
    K = K.copy()
    K[0, 0] *= 1.08
    K[1, 1] *= 1.08
    K[0, 2] += 15.0
    return K


# -- tests ----------------------------------------------------------------------------
def test_quaternions() -> None:
    """Rotation -> quaternion -> rotation is exact."""
    R = cv2.Rodrigues(np.array([0.3, -1.2, 2.0]))[0]
    err = np.abs(quat_to_mat(*mat_to_quat(R)) - R).max()
    check("quaternion round-trip", err < 1e-12, f"max matrix error {err:.1e}")


def test_board_detect() -> None:
    """Rendered boards are fully detected; a legacy board is not misread as the new layout."""
    for spec, expect in (
        ("charuco:7x5:0.035:0.026:4x4_50", 24),
        ("chessboard:9x6:0.025", 54),
        ("charuco:10x8:0.030:0.015:4x4_50:legacy", 63),
    ):
        b = Board(spec)
        img = cv2.resize(b.render(dpi=150), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        det = b.detect(img)
        n = 0 if det is None else len(det[0])
        check(f"detect rendered {spec}", n == expect, f"{n}/{expect} corners")
    legacy = Board("charuco:10x8:0.030:0.015:4x4_50:legacy").render(dpi=150)
    det = Board("charuco:10x8:0.030:0.015:4x4_50").detect(legacy)
    check(
        "legacy board rejected by new-layout spec",
        det is None,
        "no corners" if det is None else f"{len(det[0])} corners (would mislabel)",
    )


def test_intrinsics() -> None:
    """Hand-camera-like views recover K from a seed 8% off."""
    views = _synth_views(
        K_HAND, D_HAND, _random_cam_poses(40, (0.25, 0.6), 0.12), 0.2, (1280, 1056)
    )
    res = solve_intrinsics(views, (1280, 1056), _wrong_seed(K_HAND), np.zeros(5))
    dfx, dcx = abs(res["K"][0, 0] - K_HAND[0, 0]), abs(res["K"][0, 2] - K_HAND[0, 2])
    check(
        "intrinsics recovery (seed 8% off)",
        dfx < 3.0 and dcx < 3.0,
        f"{res['views']} views, fx err {dfx:.2f} px, cx err {dcx:.2f} px, "
        f"fit {res['fit_rms_px']:.3f} px, coverage {res['coverage'] * 100:.0f}%",
    )


def test_intrinsics_head_close() -> None:
    """Head-like camera (640x400), 16 close views, seeded with a wrong K."""
    views = _synth_views(K_HEAD, D_HEAD, _random_cam_poses(16, (0.15, 0.25), 0.03), 0.3, (640, 400))
    res = solve_intrinsics(views, (640, 400), _wrong_seed(K_HEAD), np.zeros(5))
    dfx = abs(res["K"][0, 0] - K_HEAD[0, 0])
    check(
        "head-like close views (seed 8% off)",
        dfx < 0.05 * K_HEAD[0, 0],
        f"{res['views']} views, fx err {dfx:.2f} px, fit {res['fit_rms_px']:.3f} px",
    )


def test_no_factory_and_camera_info() -> None:
    """No factory seed (ROS robots), then the camera_info YAML holds the result."""
    views = _synth_views(
        K_HAND, D_HAND, _random_cam_poses(40, (0.25, 0.6), 0.12), 0.2, (1280, 1056)
    )
    res = solve_intrinsics(views, (1280, 1056), None, None)
    dfx = abs(res["K"][0, 0] - K_HAND[0, 0])
    check(
        "intrinsics without factory seed",
        dfx < 3.0 and res["factory_rms_px"] is None,
        f"fx err {dfx:.2f} px, fit {res['fit_rms_px']:.3f} px",
    )
    y = camera_info_yaml("cam_test", (1280, 1056), res["K"], res["D"])

    def matrix(name: str, rows: int, cols: int) -> list[float]:
        m = re.search(rf"{name}:\n  rows: {rows}\n  cols: {cols}\n  data: \[([^\]]*)\]", y)
        return [float(v) for v in m.group(1).split(",")] if m else []

    k, p = matrix("camera_matrix", 3, 3), matrix("projection_matrix", 3, 4)
    ok = (
        len(k) == 9 and abs(k[0] - res["K"][0, 0]) < 1e-9 and len(p) == 12 and p[2] == k[2]
        and "distortion_model: plumb_bob" in y and "image_width: 1280" in y
    )  # fmt: skip
    check("camera_info YAML round-trip", ok, f"K {len(k)} values, P {len(p)} values")


def test_robot_configs() -> None:
    """Every robots/*.yaml loads; typos are rejected before capture starts."""
    names = sorted(p.stem for p in ROBOTS_DIR.glob("*.yaml"))
    cfgs = {n: load_robot(n) for n in names}
    check(
        "robot configs load",
        {"g2", "anvil"} <= set(names),
        ", ".join(f"{n}: {c['source']['type']} {len(c['cameras'])} cams" for n, c in cfgs.items()),
    )

    def rejected(mutate: Callable[[dict], Any]) -> bool:
        bad = copy.deepcopy(cfgs["anvil"])
        mutate(bad)
        try:
            validate(bad)
        except SystemExit:
            return True
        return False

    cases: dict[str, Callable[[dict], Any]] = {
        "camera without topic": lambda c: c["cameras"]["cam_chest"].pop("topic"),
        "tf stillness on rosbridge": lambda c: c["stillness"].update(type="tf"),
        "unknown source": lambda c: c["source"].update(type="usb"),
        "unknown output": lambda c: c.update(output="csv"),
        "agibot_json without vendor_intrinsic": lambda c: c.update(output="agibot_json"),
    }
    missed = [k for k, m in cases.items() if not rejected(m)]
    if rejected(lambda c: c.update(stillness={"type": "none"})):
        missed.append("valid stillness none")
    try:
        select_cams(cfgs["anvil"], "cam_chest,cam_nope")
        missed.append("unknown --cams entry")
    except SystemExit:
        pass
    detail = f"{len(cases) + 1} cases" + (f", missed {missed}" if missed else "")
    check("bad robot configs rejected", not missed, detail)


class _ScriptEnd(Exception):
    """The fake source ran out of ticks."""


class _FakeSource:
    """Scripted ticks for the capture loop, so the test never depends on timing.

    Args:
        cam: Camera name.
        frames: Images the script refers to by index.
        script: One ``(frame index or None, joint value)`` per tick.
    """

    def __init__(
        self, cam: str, frames: list[np.ndarray], script: list[tuple[int | None, float]]
    ) -> None:
        self.cam, self.script, self.last_error = cam, list(script), ""
        self.jpgs = [cv2.imencode(".jpg", f)[1].tobytes() for f in frames]
        self.grays = [
            cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_GRAYSCALE) for j in self.jpgs
        ]

    def read(self) -> Tick:
        """Next scripted tick.

        Raises:
            _ScriptEnd: When the script is used up.
        """
        if not self.script:
            raise _ScriptEnd
        i, q = self.script.pop(0)
        frame = None if i is None else Frame(self.grays[i], self.jpgs[i])
        return Tick({self.cam: frame}, np.array([q]))

    def still(self, prev: Any, cur: Any) -> bool:
        """Joint unchanged."""
        return prev is not None and float(np.max(np.abs(cur - prev))) <= 0.002

    def K_D(self, cam: str, shape: tuple[int, ...]) -> tuple[np.ndarray, None]:
        """Nominal pinhole."""
        return nominal_K(shape[1], shape[0]), None

    def sample_meta(self, tick: Tick, cams: list[str]) -> dict:
        """Joint value."""
        return {"tf": {}, "joints": [float(v) for v in tick.motion]}


def _board_frame(board: Board, size: tuple[int, int], quad: list[list[int]]) -> np.ndarray:
    """Board rendered into a grey image of ``size``, warped onto the 4-corner ``quad``."""
    src = board.render(dpi=150)
    h, w = src.shape
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(quad))
    return cv2.warpPerspective(src, H, size, borderValue=200)


def test_capture_loop() -> None:
    """The capture loop's three gates, on synthetic frames from a fake source."""
    quads = [
        [[100, 100], [500, 120], [480, 400], [90, 380]],  # view A
        [[420, 300], [860, 260], [900, 650], [450, 680]],  # view B, far from A
        [[104, 101], [503, 121], [483, 402], [93, 381]],  # A shifted 3 px
    ]
    frames = [_board_frame(BOARD, (960, 720), q) for q in quads]
    # (frame, joint). Each gate, if broken, would save at a tick with a different joint value.
    script: list[tuple[int | None, float]] = [
        (0, 0.0),  # no previous tick -> robot moving
        (1, 0.0), (0, 0.0), (1, 0.0),  # robot still, board jumps every tick -> board moving
        (1, 0.0),  # still robot, still board, new view -> SAVE B (joint 0.0)
        (1, 0.0),  # same view -> no save
        (0, 0.1), (0, 0.2), (0, 0.3),  # robot moving, board steady -> no save, board history reset
        (0, 0.3),  # board seen once since the move -> board moving
        (0, 0.3),  # -> SAVE A (joint 0.3)
        (2, 0.3),  # A shifted 3 px > 1.5 px -> board moving
        (None, 0.3),  # no frame -> no board
    ]  # fmt: skip
    args = argparse.Namespace(
        duration_s=0.0, period_s=0.0, board_still_px=1.5, min_move_m=0.03, min_rot_deg=7.0
    )
    with tempfile.TemporaryDirectory() as tmp:
        samples = Path(tmp) / "samples"
        try:
            run_loop(args, BOARD, _FakeSource("cam", frames, script), ["cam"], 10, samples)
        except _ScriptEnd:
            pass
        saved = sorted(p.name for p in samples.iterdir()) if samples.exists() else []
        joints = [json.loads((samples / s / "meta.json").read_text())["joints"] for s in saved]
        jpg_ok = all((samples / s / "cam.jpg").stat().st_size > 0 for s in saved)
    check(
        "capture loop gates (fake source)",
        saved == ["0000", "0001"] and jpg_ok and joints == [[0.0], [0.3]],
        f"saved {saved}, joints {joints}",
    )


def main() -> int:
    """Run every test; exit 1 on any failure."""
    print(f"opencv {cv2.__version__}, numpy {np.__version__}")
    test_robot_configs()
    test_capture_loop()
    test_quaternions()
    test_board_detect()
    test_intrinsics()
    test_intrinsics_head_close()
    test_no_factory_and_camera_info()
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
