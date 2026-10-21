"""AgiBot GDK environment bootstrap (G2 only).

``agibot_gdk`` links against shared objects under ``/home/agi/app/lib`` that must
be on ``LD_LIBRARY_PATH`` before the process starts: the dynamic loader reads it
at exec time, so setting it from inside Python is too late. Like every vendor
script on the robot, we re-exec the process once with the right environment.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType

APP_ROOT = "/home/agi/app"
GDK_LIB = f"{APP_ROOT}/gdk/lib"
LIB_ROOT = f"{APP_ROOT}/lib"
APP_CONF = f"{APP_ROOT}/gdk/config/app_conf.json"
_SUBDIRS = ("hal", "agibot_voice", "agora", "trro", "engines-3")
_REEXEC_FLAG = "_CAMERA_CALIB_GDK_REEXEC"


def ensure_gdk_env() -> None:
    """Put the GDK libraries on ``LD_LIBRARY_PATH``, re-exec'ing once if needed.

    Idempotent: an environment flag makes the re-exec happen at most once.
    """
    if os.environ.get(_REEXEC_FLAG) == "1":
        if GDK_LIB not in sys.path:
            sys.path.insert(0, GDK_LIB)
        return

    parts = [LIB_ROOT] + [
        os.path.join(LIB_ROOT, d) for d in _SUBDIRS if os.path.isdir(os.path.join(LIB_ROOT, d))
    ]
    if os.environ.get("LD_LIBRARY_PATH"):
        parts.append(os.environ["LD_LIBRARY_PATH"])
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    os.environ.setdefault("APP_CONF_PATH", APP_CONF)
    os.environ[_REEXEC_FLAG] = "1"

    # sys.argv[0] is the script path even under `python -m pkg.mod`; replaying it
    # would re-enter as a plain script and break relative imports, so re-exec as
    # the module __main__ was started as.
    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if main_spec is not None and main_spec.name:
        argv = [sys.executable, "-m", main_spec.name, *sys.argv[1:]]
    else:
        argv = [sys.executable, *sys.argv]
    os.execv(sys.executable, argv)


def import_gdk() -> ModuleType:
    """Return the ``agibot_gdk`` module, bootstrapping the environment first."""
    ensure_gdk_env()
    import agibot_gdk

    return agibot_gdk
