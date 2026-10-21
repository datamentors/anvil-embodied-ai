"""Offline solve: camera intrinsics from a capture run, compared with factory values.

Usage::

    python3 -m robot_camera_calib.solve RUN_DIR [--board SPEC] [--visual]

Writes ``RUN_DIR/results/``: ``report.md``, ``report.json`` and one result per
camera in the robot's output format (AgiBot ``intrinsic_*.json`` or ROS
``<cam>.yaml``). ``--visual`` adds a ``<cam>_visual.jpg`` check picture.
Nothing is installed anywhere.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .common import (
    Board,
    K_D_from_intrinsic,
    K_D_to_vendor_intrinsic,
    board_pose,
    coverage_grid,
    write_json,
)
from .config import load_robot, validate

log = logging.getLogger("camera_calib.solve")

MIN_VIEWS = 12  # fewer board views and the fit is not trusted
MIN_COVERAGE = 0.4  # fraction of the coverage grid the corners must touch


@dataclass
class View:
    """One detected board in one saved image.

    Attributes:
        sample: Sample folder name.
        obj: Nx3 board points.
        img: Nx2 detected pixels.
    """

    sample: str
    obj: np.ndarray
    img: np.ndarray


def project(
    obj: np.ndarray, T_cam_board: np.ndarray, K: np.ndarray, D: np.ndarray | None
) -> np.ndarray:
    """Project board points into the image.

    Returns:
        Nx2 pixels.
    """
    rvec = cv2.Rodrigues(T_cam_board[:3, :3])[0]
    pts, _ = cv2.projectPoints(obj, rvec, T_cam_board[:3, 3], K, D)
    return pts.reshape(-1, 2)


def reprojection_rms(
    views: Sequence[View], K: np.ndarray | None, D: np.ndarray | None
) -> float | None:
    """RMS reprojection error of a camera model on views it was not fitted to.

    Each view gets its own board pose (PnP); the error left is the model's.

    Returns:
        RMS in pixels, or None when there is no model or no view could be posed.
    """
    if K is None:
        return None
    errs = []
    for v in views:
        T = board_pose(v.obj, v.img, K, D, refine=True)
        if T is not None:
            errs.append(project(v.obj, T, K, D) - v.img)
    if not errs:
        return None
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))


def coverage(imgs: Sequence[np.ndarray], size: tuple[int, int]) -> float:
    """Fraction of the coverage grid touched by any detected corner."""
    return float(coverage_grid(list(imgs), size).mean())


def solve_intrinsics(
    views: list[View], size: tuple[int, int], K_f: np.ndarray | None, D_f: np.ndarray | None
) -> dict:
    """Fit K and D, and compare with factory values on the same frames.

    OpenCV's own initial guess (from planar homographies) is poorly conditioned
    for close, near-frontal views and can converge far away (G2 head: fx 582 vs
    factory 305), so the fit is seeded with the factory values when they exist.

    Args:
        views: Board views of one camera.
        size: Image ``(width, height)``.
        K_f: Factory camera matrix, or None (robot without factory intrinsics).
        D_f: Factory distortion, or None.

    Returns:
        ``views``, ``fit_rms_px``, ``K``, ``D``, ``std_fx_fy_cx_cy``,
        ``factory_rms_px``, ``coverage`` and, with at least 2 * MIN_VIEWS views, a
        hold-out comparison (fit on even views, test on odd): ``holdout_new_rms_px``
        and ``holdout_factory_rms_px``.
    """
    objs = [v.obj for v in views]
    imgs = [v.img for v in views]
    has_f = K_f is not None
    flags = cv2.CALIB_USE_INTRINSIC_GUESS if has_f else 0

    def seed() -> tuple[np.ndarray | None, np.ndarray | None]:
        return (K_f.copy(), D_f.copy()) if has_f else (None, None)

    rms, K, D, *rest = cv2.calibrateCameraExtended(objs, imgs, size, *seed(), flags=flags)
    out = {
        "views": len(views),
        "fit_rms_px": float(rms),
        "K": K,
        "D": D.ravel()[:5],
        "std_fx_fy_cx_cy": [float(v) for v in rest[2].ravel()[:4]],
        "factory_rms_px": reprojection_rms(views, K_f, D_f),
        "coverage": coverage(imgs, size),
    }
    if len(views) >= 2 * MIN_VIEWS:
        train, test = views[0::2], views[1::2]
        _r, K_h, D_h, *_x = cv2.calibrateCamera(
            [v.obj for v in train], [v.img for v in train], size, *seed(), flags=flags
        )
        out["holdout_new_rms_px"] = reprojection_rms(test, K_h, D_h)
        out["holdout_factory_rms_px"] = reprojection_rms(test, K_f, D_f)
    return out


def visualize(
    run: Path,
    cam: str,
    views: list[View],
    size: tuple[int, int],
    K: np.ndarray,
    D: np.ndarray,
    K_f: np.ndarray | None,
    D_f: np.ndarray | None,
    out: Path,
    height: int = 420,
) -> None:
    """Write a check picture for humans.

    Panels: detected + reprojected corners | undistorted with factory values |
    undistorted with new values | ``|factory - new|`` x5 | coverage. Both
    undistortions share one output camera, so any visible difference comes from
    the intrinsics alone. Without factory values the factory panels are left out.

    Raises:
        FileNotFoundError: If the sample image is missing.
        OSError: If the picture cannot be written.
    """
    best = max(views, key=lambda v: len(v.obj))
    img = cv2.imread(str(run / "samples" / best.sample / f"{cam}.jpg"))
    if img is None:
        raise FileNotFoundError(f"{best.sample}/{cam}.jpg")

    a = img.copy()
    T = board_pose(best.obj, best.img, K, D, refine=True)
    proj = project(best.obj, T, K, D) if T is not None else None
    err = 0.0
    for i, (x, y) in enumerate(best.img):
        cv2.circle(a, (int(x), int(y)), 4, (0, 220, 0), -1)
        if proj is not None:
            px, py = proj[i]
            cv2.circle(a, (int(px), int(py)), 2, (0, 0, 255), -1)
            tip = (int(x + 10 * (px - x)), int(y + 10 * (py - y)))
            cv2.line(a, (int(x), int(y)), tip, (0, 0, 255), 1)
    if proj is not None:
        err = float(np.sqrt(np.mean(np.sum((proj - best.img) ** 2, axis=1))))

    newK, _roi = cv2.getOptimalNewCameraMatrix(K, D, size, 0.0)
    undist_new = cv2.undistort(img, K, D, None, newK)
    w, h = size
    grid = np.full((h, w, 3), 30, np.uint8)
    for g in range(1, 8):
        cv2.line(grid, (int(g * w / 8), 0), (int(g * w / 8), h), (80, 80, 80), 1)
        cv2.line(grid, (0, int(g * h / 8)), (w, int(g * h / 8)), (80, 80, 80), 1)
    for v in views:
        for x, y in v.img:
            cv2.circle(grid, (int(x), int(y)), 2, (255, 170, 0), -1)

    items = [
        (a, f"{cam} sample {best.sample}: green detected, red reprojected (err x10) {err:.2f}px")
    ]
    if K_f is not None:
        undist_f = cv2.undistort(img, K_f, D_f, None, newK)
        items.append((undist_f, "undistorted with FACTORY intrinsics"))
    items.append((undist_new, "undistorted with NEW intrinsics"))
    if K_f is not None:
        diff = cv2.convertScaleAbs(cv2.absdiff(undist_new, undist_f), alpha=5.0)
        items.append((diff, "|factory - new| x5 (black = identical)"))
    items.append((grid, f"coverage: corners from all {len(views)} views"))

    panels = []
    for p, title in items:
        p = cv2.resize(p, (int(p.shape[1] * height / p.shape[0]), height))
        cv2.rectangle(p, (0, 0), (p.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(
            p, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
        )
        panels.append(p)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"could not write {out}")


def load_views(
    run: Path, board: Board, cams: list[str]
) -> tuple[dict[str, list[View]], dict[str, tuple[int, int]]]:
    """Detect the board in every saved image of a run.

    Returns:
        ``(views per camera, image size per camera)``.

    Raises:
        SystemExit: If the run has no samples.
    """
    views: dict[str, list[View]] = {c: [] for c in cams}
    sizes: dict[str, tuple[int, int]] = {}
    if not (run / "samples").is_dir():
        raise SystemExit(f"{run}: no samples (capture saved nothing)")
    sample_dirs = sorted(p for p in (run / "samples").iterdir() if p.is_dir())
    for sd in sample_dirs:
        for cam in cams:
            jpg = sd / f"{cam}.jpg"
            if not jpg.exists():
                continue
            gray = cv2.imread(str(jpg), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                log.warning("unreadable %s", jpg)
                continue
            sizes[cam] = (gray.shape[1], gray.shape[0])
            det = board.detect(gray)
            if det is not None:
                views[cam].append(View(sd.name, det[0], det[1]))
    log.info(
        "loaded %d samples; views per camera: %s",
        len(sample_dirs),
        {c: len(v) for c, v in views.items()},
    )
    return views, sizes


def run_config(meta: dict) -> dict:
    """The robot config a run was captured with.

    Runs made before robot configs existed have no ``robot_config`` (and G2 runs
    no ``robot`` either); for those the current ``robots/<robot>.yaml`` is used.
    """
    cfg = meta.get("robot_config")
    if cfg is None:
        return load_robot(meta.get("robot", "g2"))
    validate(cfg, "run.json robot_config")
    return cfg


def solve_run(run: Path, board: Board, visual: bool = False) -> dict:
    """Solve every camera of a run and write ``results/``.

    Args:
        run: Run folder (with ``samples/``, ``factory.json`` and ``run.json``).
        board: Calibration board.
        visual: Also write ``<cam>_visual.jpg``.

    Returns:
        The report dict (also written as ``report.json`` / ``report.md``).
    """
    factory = json.loads((run / "factory.json").read_text())
    factory_cams = factory.get("cameras", factory.get("gdk", {}))  # "gdk": runs before configs
    meta = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    cfg = run_config(meta)
    table, agibot = cfg["cameras"], cfg["output"] == "agibot_json"
    views, sizes = load_views(run, board, meta.get("cams") or list(table))
    out_dir = run / "results"
    report: dict = {"run": str(run), "board": board.spec, "robot": cfg["robot"], "cameras": {}}

    for cam, vs in views.items():
        f = factory_cams.get(cam, {})
        if "intrinsic" in f:
            K_f, D_f = K_D_from_intrinsic(f["intrinsic"], f["distortion"])
        elif agibot:
            report["cameras"][cam] = {"error": "no factory intrinsics in factory.json"}
            continue
        else:
            K_f, D_f = None, None
        r: dict = {"views": len(vs)}
        report["cameras"][cam] = r
        if len(vs) < MIN_VIEWS or cam not in sizes:
            r["intrinsics"] = {"skipped": f"need >= {MIN_VIEWS} views"}
            continue

        intr = solve_intrinsics(vs, sizes[cam], K_f, D_f)
        # A fit to these very frames must explain them at least as well as factory.
        fac_rms = intr["factory_rms_px"]
        converged = K_f is None or (fac_rms is not None and intr["fit_rms_px"] <= fac_rms + 0.02)
        ok = intr["coverage"] >= MIN_COVERAGE and converged
        if ok:
            intr["why_not"] = ""
        elif not converged:
            intr["why_not"] = "fit did not converge (worse than factory on its own frames)"
        else:
            intr["why_not"] = "poor coverage"
        r["intrinsics"] = {k: v for k, v in intr.items() if k not in ("K", "D")}
        r["intrinsics"].update(_intr_summary(intr["K"], intr["D"], K_f, D_f))
        r["intrinsics"]["recommended"] = bool(ok)
        if agibot:
            name = table[cam]["vendor_intrinsic"]
            sn = _vendor_sn(factory, name)
            write_json(out_dir / name, K_D_to_vendor_intrinsic(intr["K"], intr["D"], sn))
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{cam}.yaml").write_text(
                camera_info_yaml(cam, sizes[cam], intr["K"], intr["D"])
            )
            r["intrinsics"]["camera_info"] = f"{cam}.yaml"
        if visual:
            try:
                pic = out_dir / f"{cam}_visual.jpg"
                visualize(run, cam, vs, sizes[cam], intr["K"], intr["D"], K_f, D_f, pic)
                r["intrinsics"]["visual"] = pic.name
            except Exception as exc:  # noqa: BLE001 - a picture never fails the solve
                log.warning("%s visual failed: %s", cam, exc)

    write_json(out_dir / "report.json", report)
    (out_dir / "report.md").write_text(render_md(report))
    return report


def _intr_summary(
    K: np.ndarray, D: np.ndarray, K_f: np.ndarray | None, D_f: np.ndarray | None
) -> dict:
    names = ("fx", "fy", "cx", "cy")

    def params(K_: np.ndarray, D_: np.ndarray) -> dict:
        values = (K_[0, 0], K_[1, 1], K_[0, 2], K_[1, 2])
        out = {n: float(v) for n, v in zip(names, values, strict=True)}
        return out | {"dist": [float(v) for v in np.ravel(D_)[:5]]}

    out = {"new": params(K, D)}
    if K_f is not None:
        out["factory"] = params(K_f, D_f)
        out["delta_px"] = {n: out["new"][n] - out["factory"][n] for n in names}
    return out


def camera_info_yaml(name: str, size: tuple[int, int], K: np.ndarray, D: np.ndarray) -> str:
    """Format intrinsics as a ROS ``camera_calibration_parsers`` YAML (for ``camera_info_url``).

    Args:
        name: Camera name.
        size: Image ``(width, height)``.
        K: 3x3 camera matrix.
        D: Distortion coefficients (plumb_bob, first 5 used).

    Returns:
        YAML text.
    """
    w, h = size
    k = [float(v) for v in np.asarray(K, dtype=np.float64).ravel()]
    d = [float(v) for v in np.ravel(D)[:5]]
    p = [k[0], 0.0, k[2], 0.0, 0.0, k[4], k[5], 0.0, 0.0, 0.0, 1.0, 0.0]

    def mat(rows: int, cols: int, data: list[float]) -> str:
        return (
            f"  rows: {rows}\n  cols: {cols}\n  data: [" + ", ".join(repr(v) for v in data) + "]\n"
        )

    return (
        f"image_width: {w}\nimage_height: {h}\ncamera_name: {name}\n"
        f"camera_matrix:\n{mat(3, 3, k)}"
        f"distortion_model: plumb_bob\n"
        f"distortion_coefficients:\n{mat(1, 5, d)}"
        f"rectification_matrix:\n{mat(3, 3, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])}"
        f"projection_matrix:\n{mat(3, 4, p)}"
    )


def _vendor_sn(factory: dict, fname: str) -> str | None:
    raw = factory.get("vendor_files", {}).get(fname, "")
    try:
        return json.loads(raw).get("SN")
    except (json.JSONDecodeError, AttributeError):
        return None


def render_md(report: dict) -> str:
    """Format the report for humans.

    Args:
        report: Dict from :func:`solve_run`.

    Returns:
        Markdown text.
    """
    lines = [
        f"# {report.get('robot', 'g2').upper()} camera calibration report",
        "",
        f"- run: `{report['run']}`",
        f"- board: `{report['board']}`",
        "",
    ]
    for cam, r in report["cameras"].items():
        lines += [f"## {cam}", "", f"Board views: {r.get('views', 0)}", ""]
        if "error" in r:
            lines += [f"ERROR: {r['error']}", ""]
            continue
        i = r.get("intrinsics", {})
        if "skipped" in i:
            lines += [f"Intrinsics: skipped ({i['skipped']})", ""]
            continue
        keys = ("fx", "fy", "cx", "cy")
        has_f = "factory" in i
        lines += ["### Intrinsics", "", "| | fx | fy | cx | cy |", "|---|---|---|---|---|"]
        if has_f:
            lines.append("| factory | " + " | ".join(f"{i['factory'][k]:.2f}" for k in keys) + " |")
        lines.append("| new | " + " | ".join(f"{i['new'][k]:.2f}" for k in keys) + " |")
        if has_f:
            lines.append(
                "| delta px | " + " | ".join(f"{i['delta_px'][k]:+.2f}" for k in keys) + " |"
            )
        lines += [
            "",
            "- distortion k1 k2 p1 p2 k3: " + " ".join(f"{v:+.4f}" for v in i["new"]["dist"]),
            "- uncertainty (1 sigma) fx fy cx cy: "
            + " ".join(f"{v:.2f}" for v in i["std_fx_fy_cx_cy"])
            + " px",
        ]
        if has_f:
            lines.append(
                f"- reprojection RMS on these frames: factory {i['factory_rms_px']:.3f} px, "
                f"new (in-sample fit) {i['fit_rms_px']:.3f} px"
            )
        else:
            lines.append(
                f"- reprojection RMS on these frames (in-sample fit): {i['fit_rms_px']:.3f} px "
                "(no factory intrinsics to compare against)"
            )
        if "holdout_new_rms_px" in i:
            fac = i.get("holdout_factory_rms_px")
            fac_txt = f"factory {fac:.3f} px, " if fac is not None else ""
            lines.append(
                "- hold-out (fit on even frames, test on odd): "
                f"{fac_txt}new {i['holdout_new_rms_px']:.3f} px"
            )
        if i.get("camera_info"):
            lines.append(f"- ROS camera_info: `{i['camera_info']}`")
        lines += [
            f"- image coverage: {i['coverage'] * 100:.0f}% (need {MIN_COVERAGE * 100:.0f}%)",
            f"- recommended: {'yes' if i['recommended'] else 'NO - ' + i.get('why_not', '')}",
        ]
        if i.get("visual"):
            lines.append(f"- picture: `{i['visual']}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Parse the CLI, solve the run and print the report."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run", type=Path)
    p.add_argument(
        "--board", help="override the board spec stored in run.json (e.g. measured size)"
    )
    p.add_argument(
        "--visual", action="store_true", help="also write <cam>_visual.jpg check pictures"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    spec = args.board or json.loads((args.run / "run.json").read_text())["board"]
    solve_run(args.run, Board(spec), visual=args.visual)
    print((args.run / "results" / "report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
