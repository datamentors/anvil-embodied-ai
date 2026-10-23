"""Shared pieces: the calibration board, pose maths and file formats.

No robot SDK is imported here, so this module also runs off-robot (solve, selftest).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_BOARD = "charuco:10x8:0.030:0.015:4x4_50:legacy"
COVERAGE_CELLS = 8  # image grid used to score how much of the image the board covered

_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
}


class Board:
    """A planar calibration target, parsed from a spec string (lengths in metres).

    Specs:
        ``charuco:<squares_x>x<squares_y>:<square>:<marker>:<dict>[:legacy]``
        ``chessboard:<inner_corners_x>x<inner_corners_y>:<square>``

    ``legacy`` is the pre-OpenCV-4.6 ChArUco layout (top-left square holds a
    marker), used by calib.io and older generators. It only differs for an even
    number of rows, and a legacy board is not detected without it.

    Args:
        spec: Board spec string.

    Raises:
        ValueError: If the spec is malformed or names an unknown dictionary.
    """

    def __init__(self, spec: str) -> None:
        self.spec = spec
        parts = spec.split(":")
        self.kind = parts[0]
        nx, ny = (int(v) for v in parts[1].lower().split("x"))
        self.square = float(parts[2])
        if self.kind == "charuco":
            if len(parts) not in (5, 6) or (len(parts) == 6 and parts[5] != "legacy"):
                raise ValueError(f"bad charuco spec: {spec}")
            if parts[4] not in _DICTS:
                raise ValueError(f"unknown dict {parts[4]}; known: {sorted(_DICTS)}")
            dictionary = cv2.aruco.getPredefinedDictionary(_DICTS[parts[4]])
            self.squares = (nx, ny)
            self._board = cv2.aruco.CharucoBoard((nx, ny), self.square, float(parts[3]), dictionary)
            self._board.setLegacyPattern(len(parts) == 6)
            # The defaults miss soft, blurred markers at the edges of wide lenses
            # (G2 hand camera, lower-left corner: 8.8 -> 38.3 of 63 corners with these).
            dp = cv2.aruco.DetectorParameters()
            dp.perspectiveRemovePixelPerCell = 8
            dp.perspectiveRemoveIgnoredMarginPerCell = 0.2
            dp.adaptiveThreshWinSizeMin = 3
            dp.adaptiveThreshWinSizeMax = 53
            dp.adaptiveThreshWinSizeStep = 6
            cp = cv2.aruco.CharucoParameters()
            cp.tryRefineMarkers = True  # recover missed markers from the board layout
            self._detector = cv2.aruco.CharucoDetector(self._board, cp, dp)
            self.min_points = 6
        elif self.kind == "chessboard":
            if len(parts) != 3:
                raise ValueError(f"bad chessboard spec: {spec}")
            self.inner = (nx, ny)
            self.squares = (nx + 1, ny + 1)
            grid = np.zeros((nx * ny, 3), np.float32)
            grid[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * self.square
            self._grid = grid
            self.min_points = nx * ny
        else:
            raise ValueError(f"unknown board kind {self.kind!r} (charuco|chessboard)")

    def detect(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Find the board in a grayscale image.

        Args:
            gray: Grayscale uint8 image.

        Returns:
            ``(object_points Nx3, image_points Nx2)`` as float32, or None when too
            few corners were found or they all lie on one line (no pose).
        """
        if self.kind == "charuco":
            cc, ci, _mc, _mi = self._detector.detectBoard(gray)
            if ci is None or len(ci) < self.min_points:
                return None
            obj, img = self._board.matchImagePoints(cc, ci)
            obj, img = obj.reshape(-1, 3), img.reshape(-1, 2)
        else:
            flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            ok, corners = cv2.findChessboardCornersSB(gray, self.inner, flags=flags)
            if not ok:
                return None
            obj, img = self._grid, corners.reshape(-1, 2)
        if np.linalg.matrix_rank(obj[:, :2] - obj[:, :2].mean(0), tol=1e-6) < 2:
            return None
        return obj.astype(np.float32), img.astype(np.float32)

    def render(self, dpi: int = 300, margin_mm: float = 10.0) -> np.ndarray:
        """Draw the board at true scale for printing.

        Args:
            dpi: Print resolution.
            margin_mm: White margin around the board.

        Returns:
            Grayscale uint8 image.
        """
        px_per_m = dpi / 0.0254
        sq = int(round(self.square * px_per_m))
        margin = int(round(margin_mm / 1000 * px_per_m))
        w, h = self.squares[0] * sq + 2 * margin, self.squares[1] * sq + 2 * margin
        if self.kind == "charuco":
            return self._board.generateImage((w, h), marginSize=margin, borderBits=1)
        img = np.full((h, w), 255, np.uint8)
        for y in range(self.squares[1]):
            for x in range(self.squares[0]):
                if (x + y) % 2 == 0:
                    y0, x0 = margin + y * sq, margin + x * sq
                    img[y0 : y0 + sq, x0 : x0 + sq] = 0
        return img


# -- poses (4x4 matrices, quaternions x, y, z, w) ------------------------------------
def make_T(R: np.ndarray, t: Any) -> np.ndarray:
    """Build a 4x4 rigid transform from a 3x3 rotation and a translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def quat_to_mat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert a quaternion (any norm) to a 3x3 rotation matrix.

    Raises:
        ValueError: If the quaternion is all zeros.
    """
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("zero quaternion")
    x, y, z, w = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(R: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to a unit quaternion ``[x, y, z, w]`` with w >= 0."""
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x, y, z = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x, y, z = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x, y, z = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x, y, z = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    q = np.array([x, y, z, w])
    if q[3] < 0:
        q = -q
    return [float(v) for v in q / np.linalg.norm(q)]


def T_to_pose7(T: np.ndarray) -> list[float]:
    """Convert a 4x4 transform to ``[x, y, z, qx, qy, qz, qw]``."""
    return [float(v) for v in T[:3, 3]] + mat_to_quat(T[:3, :3])


def pose_delta(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """Distance between two 4x4 poses.

    Returns:
        ``(translation in metres, rotation in degrees)``.
    """
    D = np.linalg.inv(A) @ B
    cos = (np.trace(D[:3, :3]) - 1.0) / 2.0
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3])), float(
        np.degrees(np.arccos(np.clip(cos, -1, 1)))
    )


def board_pose(
    obj: np.ndarray, img: np.ndarray, K: np.ndarray, D: np.ndarray | None, refine: bool = False
) -> np.ndarray | None:
    """Pose of the board in the camera frame (camera -> board) from its detected corners.

    Args:
        obj: Nx3 board points.
        img: Nx2 detected pixels.
        K: 3x3 camera matrix.
        D: Distortion coefficients, or None for none.
        refine: Polish the planar (IPPE) solution with Levenberg-Marquardt.

    Returns:
        4x4 transform, or None if PnP failed.
    """
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    if refine:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
    return make_T(cv2.Rodrigues(rvec)[0], tvec)


def coverage_grid(points: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """Mark which cells of a COVERAGE_CELLS x COVERAGE_CELLS image grid hold a corner.

    Args:
        points: Nx2 pixel arrays, one per view.
        size: Image ``(width, height)``.

    Returns:
        Boolean ``(rows, cols)`` array; its mean is the coverage fraction.
    """
    w, h = size
    hit = np.zeros((COVERAGE_CELLS, COVERAGE_CELLS), bool)
    for pts in points:
        xs = np.clip((pts[:, 0] / w * COVERAGE_CELLS).astype(int), 0, COVERAGE_CELLS - 1)
        ys = np.clip((pts[:, 1] / h * COVERAGE_CELLS).astype(int), 0, COVERAGE_CELLS - 1)
        hit[ys, xs] = True
    return hit


# -- file formats ----------------------------------------------------------------------
def K_D_from_intrinsic(intr: list[float], dist: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert GDK ``[fx, fy, cx, cy]`` and ``[k1, k2, p1, p2, k3, ...]`` to OpenCV K, D(5)."""
    fx, fy, cx, cy = intr[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(5)
    D[: min(5, len(dist))] = dist[:5]
    return K, D


def K_D_to_vendor_intrinsic(K: np.ndarray, D: np.ndarray, sn: str | None) -> dict:
    """Convert OpenCV K, D to AgiBot's ``intrinsic_*.json`` layout.

    Args:
        K: 3x3 camera matrix.
        D: Distortion coefficients (first 5 used).
        sn: Camera serial number to keep, or None.

    Returns:
        Dict with Fx, Fy, Cx, Cy, k1, k2, p1, p2, k3 and optionally SN.
    """
    d = np.zeros(5)
    flat = np.asarray(D, dtype=np.float64).ravel()
    d[: min(5, flat.size)] = flat[:5]
    out = {
        "Fx": float(K[0, 0]), "Fy": float(K[1, 1]), "Cx": float(K[0, 2]), "Cy": float(K[1, 2]),
        "k1": float(d[0]), "k2": float(d[1]), "p1": float(d[2]), "p2": float(d[3]),
        "k3": float(d[4]),
    }  # fmt: skip
    if sn is not None:
        out["SN"] = sn
    return out


def write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically (temp file, then rename), creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)
