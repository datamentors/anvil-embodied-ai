"""Tests for the annotate-dataset CLI (labels written into a converted dataset)."""
from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from mcap_converter.cli import annotate_dataset
from mcap_converter.cli.annotate_dataset import is_label_session_record, spread_session_labels
from mcap_converter.cli.label_session import (
    DEFAULT_SORTING_RULES,
    DEFAULT_TASK,
    build_session_metadata,
)


@pytest.fixture
def raw_session(tmp_path):
    """A raw session labelled big/downside: three recordings, the second aborted."""
    root = tmp_path / "raw" / "2026-08-24-s01"
    for name, status in (("0001", "success"), ("0002", "aborted"), ("0003", "success")):
        d = root / name
        d.mkdir(parents=True)
        (d / f"{name}_0.mcap").write_bytes(b"not really an mcap")
        (d / "metadata.json").write_text(json.dumps({"version": 1, "status": status}))
    episodes = sorted(p for p in root.iterdir() if p.is_dir())
    record = build_session_metadata(
        root, episodes, "big", "downside", None, DEFAULT_TASK, DEFAULT_SORTING_RULES
    )
    (root / "session_metadata.json").write_text(json.dumps(record))
    return root


def make_dataset(root, n_episodes):
    """The parts of a LeRobot dataset annotate-dataset reads and writes."""
    episodes_dir = root / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": n_episodes}))
    pd.DataFrame({"episode_index": range(n_episodes), "length": [10] * n_episodes}).to_parquet(
        episodes_dir / "file-000.parquet"
    )
    return root


def read_episodes(root):
    return pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["annotate-dataset", *map(str, argv)])
    annotate_dataset.main()


def label_session_entries(raw_session):
    return json.loads((raw_session / "session_metadata.json").read_text())["episodes"]


# =============================================================================
# recognising a label-session record
# =============================================================================


class TestIsLabelSessionRecord:
    def test_label_session_entries_are_recognised(self, raw_session):
        assert is_label_session_record(label_session_entries(raw_session))

    def test_audit_entries_are_not(self):
        entries = [{"episode_index": 0, "envelope_size": "big"}]
        assert not is_label_session_record(entries)


# =============================================================================
# spreading the session envelope over the dataset
# =============================================================================


class TestSpreadSessionLabels:
    def test_every_episode_gets_the_session_envelope(self, raw_session):
        by_index = spread_session_labels(label_session_entries(raw_session), 3, raw_session)
        assert sorted(by_index) == [0, 1, 2]
        for entry in by_index.values():
            assert entry == {
                "envelope_size": "big",
                "envelope_facing_side": "downside",
                "destination_basket_side": "left",
            }

    def test_per_recording_fields_stay_out(self, raw_session):
        by_index = spread_session_labels(label_session_entries(raw_session), 3, raw_session)
        for entry in by_index.values():
            assert not {"episode_dir", "recorder_status", "recorder_note"} & set(entry)

    def test_fewer_dataset_episodes_than_recordings_is_fine(self, raw_session):
        # The converter skipped the aborted recording.
        by_index = spread_session_labels(label_session_entries(raw_session), 2, raw_session)
        assert sorted(by_index) == [0, 1]

    def test_more_dataset_episodes_than_recordings_is_refused(self, raw_session):
        with pytest.raises(ValueError, match="lists only 3 recordings"):
            spread_session_labels(label_session_entries(raw_session), 4, raw_session)

    def test_differing_envelopes_are_refused(self, raw_session):
        entries = label_session_entries(raw_session)
        entries[2]["envelope_size"] = "small"
        with pytest.raises(ValueError, match=r"\['0003'\] carry a different envelope"):
            spread_session_labels(entries, 3, raw_session)


# =============================================================================
# CLI
# =============================================================================


class TestMainWithLabelSessionRecord:
    def test_writes_the_envelope_on_every_row(self, monkeypatch, tmp_path, raw_session):
        dataset = make_dataset(tmp_path / "dataset", 2)
        run_cli(monkeypatch, dataset, "--metadata", raw_session / "session_metadata.json")

        df = read_episodes(dataset)
        assert list(df["envelope_size"]) == ["big", "big"]
        assert list(df["envelope_facing_side"]) == ["downside", "downside"]
        assert "destination_basket_side" not in df.columns
        assert list(df["length"]) == [10, 10]

    def test_include_derived_adds_the_basket(self, monkeypatch, tmp_path, raw_session):
        dataset = make_dataset(tmp_path / "dataset", 2)
        run_cli(
            monkeypatch, dataset,
            "--metadata", raw_session / "session_metadata.json", "--include-derived",
        )
        assert list(read_episodes(dataset)["destination_basket_side"]) == ["left", "left"]

    def test_dry_run_writes_nothing(self, monkeypatch, tmp_path, raw_session):
        dataset = make_dataset(tmp_path / "dataset", 2)
        run_cli(
            monkeypatch, dataset, "--metadata", raw_session / "session_metadata.json", "--dry-run"
        )
        assert "envelope_size" not in read_episodes(dataset).columns

    def test_dataset_larger_than_session_exits(self, monkeypatch, tmp_path, raw_session):
        dataset = make_dataset(tmp_path / "dataset", 5)
        with pytest.raises(SystemExit):
            run_cli(monkeypatch, dataset, "--metadata", raw_session / "session_metadata.json")
        assert "envelope_size" not in read_episodes(dataset).columns


class TestMainWithAuditRecord:
    def test_entries_are_placed_by_episode_index(self, monkeypatch, tmp_path):
        dataset = make_dataset(tmp_path / "dataset", 2)
        audit = tmp_path / "audit.json"
        audit.write_text(json.dumps({"loop": {"episodes": [
            {"episode_index": 1, "envelope_size": "small", "envelope_facing_side": "upside"},
            {"episode_index": 0, "envelope_size": "big", "envelope_facing_side": "downside"},
        ]}}))
        run_cli(monkeypatch, dataset, "--metadata", audit)
        assert list(read_episodes(dataset)["envelope_size"]) == ["big", "small"]
