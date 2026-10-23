"""Robot configs: one YAML per robot under ``robots/``.

A config says where frames come from (``source``), how "robot still" is judged
(``stillness``), which cameras exist and in which format results are written
(``output``). See README.md in this folder for every key.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROBOTS_DIR = Path(__file__).resolve().parent.parent / "robots"

# source type -> (stillness types it can provide, keys every camera needs)
SOURCES = {
    "gdk": ({"tf", "none"}, ("gdk", "parent")),
    "rosbridge": ({"joint_states", "none"}, ("topic",)),
}
# output type -> keys every camera needs
OUTPUTS = {"agibot_json": ("vendor_intrinsic",), "ros_camera_info": ()}
# stillness type -> keys it needs
# "none": fixed cameras / no motion feed; the board-still gate still rejects blur.
STILLNESS = {"tf": ("still_mm", "still_deg"), "joint_states": ("topic", "still_rad"), "none": ()}


def resolve(name_or_path: str) -> Path:
    """Turn ``--robot`` into a file path.

    Args:
        name_or_path: A robot name (``robots/<name>.yaml``) or a path to a YAML file.

    Returns:
        Path of the config file.
    """
    p = Path(name_or_path).expanduser()
    if p.suffix in (".yaml", ".yml"):
        return p
    return ROBOTS_DIR / f"{name_or_path}.yaml"


def load_robot(name_or_path: str) -> dict:
    """Load and validate a robot config.

    Args:
        name_or_path: A robot name or a path to a YAML file.

    Returns:
        The config dict, plus ``config_path``.

    Raises:
        SystemExit: If the file is missing, is not YAML, or fails validation.
    """
    path = resolve(name_or_path)
    try:
        cfg = yaml.safe_load(path.read_text())
    except OSError as exc:
        known = sorted(p.stem for p in ROBOTS_DIR.glob("*.yaml"))
        raise SystemExit(f"robot config {path}: {exc} (known robots: {known})") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"robot config {path}: invalid YAML: {exc}") from exc
    validate(cfg, str(path))
    cfg["config_path"] = str(path)
    return cfg


def validate(cfg: object, where: str = "robot config") -> None:
    """Check a config so that a typo fails before capture starts.

    Args:
        cfg: Parsed YAML.
        where: Name used in the error message.

    Raises:
        SystemExit: Naming the first problem found.
    """

    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise SystemExit(f"{where}: {msg}")

    need(isinstance(cfg, dict), "top level must be a mapping")
    assert isinstance(cfg, dict)
    need(isinstance(cfg.get("robot"), str) and cfg["robot"] != "", "`robot` (name) is required")
    src, still = cfg.get("source"), cfg.get("stillness")
    need(
        isinstance(src, dict) and src.get("type") in SOURCES,
        f"`source.type` must be one of {sorted(SOURCES)}",
    )
    stills, cam_keys = SOURCES[src["type"]]
    need(
        isinstance(still, dict) and still.get("type") in stills,
        f"`stillness.type` for source {src['type']} must be one of {sorted(stills)}",
    )
    for k in STILLNESS[still["type"]]:
        need(still.get(k) is not None, f"`stillness.{k}` is required for stillness {still['type']}")
    if src["type"] == "rosbridge":
        need(isinstance(src.get("url"), str), "`source.url` (rosbridge websocket) is required")
    need(cfg.get("output") in OUTPUTS, f"`output` must be one of {sorted(OUTPUTS)}")
    need(isinstance(cfg.get("runs_dir"), str), "`runs_dir` is required")
    cams = cfg.get("cameras")
    need(isinstance(cams, dict) and len(cams) > 0, "`cameras` must map camera name -> settings")
    for name, c in cams.items():
        need(isinstance(c, dict), f"camera {name}: settings must be a mapping")
        for k in cam_keys + OUTPUTS[cfg["output"]]:
            need(c.get(k) is not None, f"camera {name}: `{k}` is required")


def runs_path(cfg: dict) -> Path:
    """Where runs go: ``runs_dir`` as given, or relative to the package folder."""
    p = Path(cfg["runs_dir"]).expanduser()
    return p if p.is_absolute() else ROBOTS_DIR.parent / p


def select_cams(cfg: dict, cams_arg: str | None) -> list[str]:
    """Pick the cameras to capture.

    Args:
        cfg: Robot config.
        cams_arg: ``--cams`` value (comma list), or None for every configured camera.

    Returns:
        Camera names, in the order given.

    Raises:
        SystemExit: If a name is not in the config.
    """
    if not cams_arg:
        return list(cfg["cameras"])
    cams = [c.strip() for c in cams_arg.split(",") if c.strip()]
    unknown = [c for c in cams if c not in cfg["cameras"]]
    if unknown:
        raise SystemExit(f"unknown cameras {unknown}; {cfg['robot']} has: {list(cfg['cameras'])}")
    return cams
