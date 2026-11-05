"""Git provenance helpers — stamp artifact files with the producing commit/tag."""
from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]  # …/anvil-embodied-ai


def git_provenance() -> dict[str, str]:
    """Return the git commit SHA and tag for the current repo HEAD.

    Returns a dict with zero, one, or both of:
        ``code_commit`` — full SHA of HEAD
        ``code_tag``    — tag name when HEAD sits exactly on a tag

    Outside a git checkout (an installed wheel) it returns ``code_version``
    instead: the installed anvil-shared version, which the bundle workflow
    builds as ``<version>+g<short sha>``.

    All subprocess errors are swallowed; callers receive an empty dict
    rather than a crash when git is unavailable or the directory is not
    a repository and no installed version is found.
    """
    result: dict[str, str] = {}
    try:
        result["code_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        pass
    try:
        result["code_tag"] = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        pass
    if "code_commit" not in result:
        try:
            result["code_version"] = metadata.version("anvil-shared")
        except metadata.PackageNotFoundError:
            pass
    return result
