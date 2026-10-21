"""Passive calibration capture for any configured robot (``--robot``, see robots/).

Read-only: it only watches. Whoever is teleoperating keeps full control.

Every tick (``--period-s``) it reads all cameras, and saves a sample for the
cameras where:

1. the robot was still across two ticks,
2. the board did not move in that camera since the previous tick (no blur), and
3. the board view differs enough from the views already saved for that camera.

Usage::

    python3 -m robot_camera_calib.capture --robot g2 --run intr01 --board <spec> --web
    ./run.sh capture --robot anvil --run anvil01 --board <spec> --web
    python3 -m robot_camera_calib.capture --print-board board.png
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import DEFAULT_BOARD, Board, board_pose, pose_delta, write_json
from .config import load_robot, runs_path, select_cams
from .sources import Source, Tick

log = logging.getLogger("camera_calib.capture")

# CLI flag -> (config section, the section type it applies to)
_OVERRIDES = {
    "still_mm": ("stillness", "tf"),
    "still_deg": ("stillness", "tf"),
    "still_rad": ("stillness", "joint_states"),
    "image_timeout_ms": ("source", "gdk"),
    "throttle_ms": ("source", "rosbridge"),
    "rosbridge": ("source", "rosbridge"),
}


class _Stop:
    """Turn Ctrl-C / SIGTERM into a flag so the loop finishes its tick cleanly."""

    def __init__(self) -> None:
        self.flag = False
        signal.signal(signal.SIGINT, self._set)
        signal.signal(signal.SIGTERM, self._set)

    def _set(self, *_a: Any) -> None:
        self.flag = True


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI flags over the robot config.

    Args:
        cfg: Robot config.
        args: Parsed CLI arguments.

    Returns:
        The updated config.

    Raises:
        SystemExit: If a flag does not apply to this robot's source or stillness type.
    """
    for key, (section, kind) in _OVERRIDES.items():
        value = getattr(args, key)
        if value is None:
            continue
        if cfg[section]["type"] != kind:
            raise SystemExit(
                f"--{key.replace('_', '-')} needs {section} type {kind}; "
                f"{cfg['robot']} uses {cfg[section]['type']}"
            )
        cfg[section]["url" if key == "rosbridge" else key] = value
    for key in ("runs_dir", "max_samples"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    cfg.setdefault("max_samples", 80)
    return cfg


def capture(args: argparse.Namespace) -> int:
    """Run one capture: factory dump, run.json, then the loop.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    from .sources import make_source

    cfg = apply_overrides(load_robot(args.robot), args)
    board = Board(args.board)
    cams = select_cams(cfg, args.cams)
    run_dir = runs_path(cfg) / args.run
    samples_dir = run_dir / "samples"
    if samples_dir.exists() and any(samples_dir.iterdir()) and not args.factory_only:
        raise SystemExit(f"{samples_dir} already has samples; pick a new --run name")

    source = make_source(cfg, cams)
    try:
        source.write_factory(run_dir)
        if args.factory_only:
            return 0
        write_json(
            run_dir / "run.json",
            {
                "robot": cfg["robot"], "output": cfg["output"], "board": args.board, "cams": cams,
                "host": socket.gethostname(), "started_unix": time.time(), **source.run_info(),
                "board_still_px": args.board_still_px, "min_move_m": args.min_move_m,
                "min_rot_deg": args.min_rot_deg, "robot_config": cfg,
            },
        )  # fmt: skip
        view = None
        if args.web:
            from .web import LiveView

            view = LiveView(cams, args.web_host, args.web_port)
            view.set_status(run=args.run, board=args.board)
        try:
            return run_loop(args, board, source, cams, int(cfg["max_samples"]), samples_dir, view)
        finally:
            if view is not None:
                _linger(args.web_linger_s)
                view.close()
    finally:
        source.close()


def _linger(seconds: float) -> None:
    """Keep the live view up after capture ends, until timeout or Ctrl-C."""
    if seconds <= 0:
        return
    signal.signal(signal.SIGINT, signal.default_int_handler)  # _Stop swallowed it
    log.info("capture finished; live view stays up %.0f s (Ctrl-C to quit now)", seconds)
    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        pass


def run_loop(
    args: argparse.Namespace,
    board: Board,
    source: Source,
    cams: list[str],
    max_samples: int,
    samples_dir: Path,
    view: Any = None,
) -> int:
    """Watch the cameras and save samples until stopped, timed out or full.

    Args:
        args: Uses ``duration_s``, ``period_s``, ``board_still_px``, ``min_move_m``
            and ``min_rot_deg``.
        board: Calibration board.
        source: Frame source.
        cams: Cameras to watch.
        max_samples: Stop after this many samples.
        samples_dir: Where ``NNNN/`` sample folders go.
        view: Optional live view (``web.LiveView``).

    Returns:
        0.
    """
    stop = _Stop()
    saved_views: dict[str, list[np.ndarray]] = {c: [] for c in cams}
    prev_board: dict[str, tuple] = {}  # cam -> detection on the previous tick
    prev_motion: Any = None
    n_saved, last_report, last_warn = 0, 0.0, 0.0
    counts = dict.fromkeys(
        ("ticks", "moving", "board_moving", "no_board", "same_view", "no_data"), 0
    )
    t_end = time.monotonic() + args.duration_s if args.duration_s > 0 else None
    log.info("watching %s for board %s -- Ctrl-C to stop", cams, board.spec)

    while not stop.flag and n_saved < max_samples:
        if t_end is not None and time.monotonic() > t_end:
            break
        tick_t = time.monotonic()
        counts["ticks"] += 1
        tick = source.read()
        if tick is None:
            counts["no_data"] += 1
            prev_motion = None  # a gap breaks the "still across two ticks" chain
            if tick_t - last_warn > 5.0:
                last_warn = tick_t
                log.warning("%s", source.last_error)
            _sleep_rest(tick_t, args.period_s)
            continue

        grays = {c: (f.gray if f is not None else None) for c, f in tick.frames.items()}
        still = source.still(prev_motion, tick.motion)
        prev_motion = tick.motion
        states = dict.fromkeys(cams, "robot moving")
        pts: dict[str, np.ndarray] = {}
        if not still:
            counts["moving"] += 1
            prev_board = {}
        else:
            detections, new_view, board_now = {}, [], {}
            for c in cams:
                states[c] = _check_camera(
                    c, grays[c], board, source, args, prev_board, board_now, saved_views, pts
                )
                if states[c] == "board moving - hold still":
                    counts["board_moving"] += 1
                if states[c] in ("new view", "same view - move the board somewhere new"):
                    detections[c] = len(pts[c])
                if states[c] == "new view":
                    new_view.append(c)
            prev_board = board_now
            if not detections:
                counts["no_board"] += 1
            elif not new_view:
                counts["same_view"] += 1
            else:
                _save(
                    samples_dir / f"{n_saved:04d}",
                    tick,
                    list(detections),
                    detections,
                    new_view,
                    source,
                )
                n_saved += 1
                log.info("saved #%d  new view: %s  board seen: %s", n_saved, new_view, detections)
                for c in detections:  # a sample holds every camera that passed gate 2
                    states[c] = "SAVED"
                    if view is not None:
                        view.add_saved(c, pts[c])

        if view is not None:
            for c in cams:
                view.update(c, grays[c], states[c], pts.get(c))
            view.set_status(counts=counts, saved=n_saved)
        if tick_t - last_report > 10.0:
            last_report = tick_t
            log.info("status: saved=%d views=%s %s", n_saved, _per_cam(saved_views), counts)
        _sleep_rest(tick_t, args.period_s)

    log.info("done: saved=%d views=%s %s", n_saved, _per_cam(saved_views), counts)
    if view is not None:
        view.set_status(done=True)
    return 0


def _check_camera(
    cam: str,
    gray: np.ndarray | None,
    board: Board,
    source: Source,
    args: argparse.Namespace,
    prev_board: dict,
    board_now: dict,
    saved_views: dict[str, list[np.ndarray]],
    pts: dict[str, np.ndarray],
) -> str:
    """Apply gates 2 and 3 to one camera on a still tick.

    Records the detection in ``board_now`` / ``pts`` and, for a new view, the
    board pose in ``saved_views``.

    Returns:
        ``"no board"``, ``"board moving - hold still"``,
        ``"same view - move the board somewhere new"`` or ``"new view"``.
    """
    if gray is None or (det := board.detect(gray)) is None:
        return "no board"
    pts[cam] = det[1]
    K, D = source.K_D(cam, gray.shape)
    T_cam_board = board_pose(det[0], det[1], K, D)
    if T_cam_board is None:
        return "no board"
    board_now[cam] = det
    # Pixels, not board pose: a partial board at the image edge gives a jittery
    # pose even when it is held still.
    if cam not in prev_board or not _board_still(prev_board[cam], det, args.board_still_px):
        return "board moving - hold still"
    if not all(_far(T_cam_board, v, args) for v in saved_views[cam]):
        return "same view - move the board somewhere new"
    saved_views[cam].append(T_cam_board)
    return "new view"


def _board_still(prev_det: tuple, det: tuple, max_px: float, min_common: int = 4) -> bool:
    """Median pixel shift of the corners seen on both ticks is at most ``max_px``."""
    prev = {tuple(np.round(o, 5)): p for o, p in zip(prev_det[0], prev_det[1], strict=True)}
    shifts = [
        np.linalg.norm(p - prev[k])
        for o, p in zip(det[0], det[1], strict=True)
        if (k := tuple(np.round(o, 5))) in prev
    ]
    return len(shifts) >= min_common and float(np.median(shifts)) <= max_px


def _far(A: np.ndarray, B: np.ndarray, args: argparse.Namespace) -> bool:
    dt, dr = pose_delta(A, B)
    return dt >= args.min_move_m or dr >= args.min_rot_deg


def _per_cam(saved_views: dict[str, list[np.ndarray]]) -> dict[str, int]:
    return {c: len(v) for c, v in saved_views.items()}


def _save(
    d: Path, tick: Tick, cams: list[str], detections: dict, new_view: list[str], source: Source
) -> None:
    d.mkdir(parents=True, exist_ok=False)
    for c in cams:
        (d / f"{c}.jpg").write_bytes(tick.frames[c].jpg)
    meta = {"wall_time": time.time(), **source.sample_meta(tick, cams)}
    write_json(d / "meta.json", {**meta, "detected": detections, "new_view": new_view})


def _sleep_rest(tick_t: float, period: float) -> None:
    rest = period - (time.monotonic() - tick_t)
    if rest > 0:
        time.sleep(rest)


def main() -> int:
    """Parse the CLI and run capture (or print a board)."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--robot", default="g2", help="robots/<name>.yaml or a .yaml path (default %(default)s)"
    )
    p.add_argument("--run", help="run name (folder under the runs dir)")
    p.add_argument("--runs-dir", default=None, help="override the config's runs_dir")
    p.add_argument(
        "--board", default=DEFAULT_BOARD, help="MEASURED board spec (default %(default)s)"
    )
    p.add_argument("--cams", default=None, help="comma list, subset of the config's cameras")
    p.add_argument("--factory-only", action="store_true", help="write factory.json and exit")
    p.add_argument("--print-board", metavar="PNG", help="write a printable 300 dpi board and exit")
    p.add_argument(
        "--max-samples", type=int, default=None, help="override the config's max_samples"
    )
    p.add_argument(
        "--duration-s", type=float, default=0.0, help="stop after N s (0 = until Ctrl-C)"
    )
    p.add_argument("--period-s", type=float, default=0.3, help="tick length")
    p.add_argument(
        "--still-mm", type=float, default=None, help="tf stillness: max frame motion per tick"
    )
    p.add_argument("--still-deg", type=float, default=None, help="tf stillness: max frame rotation")
    p.add_argument(
        "--still-rad", type=float, default=None, help="joint_states stillness: max change"
    )
    p.add_argument("--image-timeout-ms", type=float, default=None, help="gdk: image read timeout")
    p.add_argument("--rosbridge", default=None, help="rosbridge: override source.url")
    p.add_argument(
        "--throttle-ms", type=int, default=None, help="rosbridge: per-camera frame interval"
    )
    p.add_argument(
        "--board-still-px", type=float, default=1.5, help="max median corner shift per tick"
    )
    p.add_argument(
        "--min-move-m", type=float, default=0.03, help="new view if the board moved this much"
    )
    p.add_argument("--min-rot-deg", type=float, default=7.0, help="... or rotated this much")
    p.add_argument(
        "--web", action="store_true", help="serve a live view (./sync.sh view on the Mac)"
    )
    p.add_argument("--web-host", default="127.0.0.1", help="0.0.0.0 = whole LAN, no auth")
    p.add_argument("--web-port", type=int, default=9701)
    p.add_argument("--web-linger-s", type=float, default=0.0, help="keep the view up after capture")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.print_board:
        b = Board(args.board)
        img = b.render()
        if not cv2.imwrite(args.print_board, img):
            raise SystemExit(f"could not write {args.print_board}")
        w_mm, h_mm = img.shape[1] / 300 * 25.4, img.shape[0] / 300 * 25.4
        print(
            f"wrote {args.print_board} ({w_mm:.0f}x{h_mm:.0f} mm at 300 dpi). Print at 100%, then "
            f"measure one square edge (nominal {b.square * 1000:.1f} mm) and use it in --board."
        )
        return 0
    if not args.run:
        p.error("--run is required")
    os.umask(0o022)
    return capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
