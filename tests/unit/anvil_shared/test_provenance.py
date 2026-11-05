"""Tests for anvil_shared.provenance."""

from __future__ import annotations

import subprocess
from importlib import metadata

from anvil_shared import provenance


def _no_git(*args, **kwargs):
    raise subprocess.CalledProcessError(128, args[0])


def test_inside_a_checkout_reports_the_commit(monkeypatch):
    monkeypatch.setattr(provenance.subprocess, "check_output", lambda *_args, **_kwargs: "abc123\n")
    result = provenance.git_provenance()
    assert result["code_commit"] == "abc123"
    assert "code_version" not in result


def test_outside_a_checkout_falls_back_to_the_installed_version(monkeypatch):
    monkeypatch.setattr(provenance.subprocess, "check_output", _no_git)
    monkeypatch.setattr(provenance.metadata, "version", lambda _name: "0.1.0+gb300789")
    assert provenance.git_provenance() == {"code_version": "0.1.0+gb300789"}


def test_nothing_known_returns_empty(monkeypatch):
    def not_installed(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(provenance.subprocess, "check_output", _no_git)
    monkeypatch.setattr(provenance.metadata, "version", not_installed)
    assert provenance.git_provenance() == {}
