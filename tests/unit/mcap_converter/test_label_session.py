"""Tests for the label-session CLI (data-collection session labelling)."""
from __future__ import annotations

import json

import pytest

from mcap_converter.cli.label_session import (
    DEFAULT_SORTING_RULES,
    DEFAULT_TASK,
    build_session_metadata,
    find_conflicts,
    find_episodes,
    read_sidecar,
    stamp_episode,
)


@pytest.fixture
def session(tmp_path):
    """A session folder with three episodes, one of them aborted."""
    root = tmp_path / "2026-08-24-s01"
    for name, status in (("0001", "success"), ("0002", "aborted"), ("0003", "success")):
        d = root / name
        d.mkdir(parents=True)
        (d / f"{name}_0.mcap").write_bytes(b"not really an mcap")
        (d / "metadata.json").write_text(
            json.dumps({"version": 1, "status": status, "note": None, "duration": 4})
        )
    return root


# =============================================================================
# discovery
# =============================================================================


class TestFindEpisodes:
    def test_finds_episode_dirs_in_order(self, session):
        assert [d.name for d in find_episodes(session)] == ["0001", "0002", "0003"]

    def test_ignores_dirs_without_mcap(self, session):
        (session / "notes").mkdir()
        (session / "notes" / "readme.txt").write_text("hello")
        assert [d.name for d in find_episodes(session)] == ["0001", "0002", "0003"]

    def test_missing_folder(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_episodes(tmp_path / "nope")

    def test_path_that_is_a_file(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            find_episodes(f)

    def test_folder_with_no_episodes_says_so(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="no episodes found"):
            find_episodes(empty)


class TestReadSidecar:
    def test_reads_existing(self, session):
        assert read_sidecar(session / "0001")["status"] == "success"

    def test_missing_sidecar_is_empty_dict(self, tmp_path):
        d = tmp_path / "0009"
        d.mkdir()
        assert read_sidecar(d) == {}

    def test_corrupt_sidecar_is_empty_dict(self, tmp_path):
        d = tmp_path / "0009"
        d.mkdir()
        (d / "metadata.json").write_text("{ not json")
        assert read_sidecar(d) == {}


# =============================================================================
# session record
# =============================================================================


class TestBuildSessionMetadata:
    def _build(self, session, size="big", face="upside", arm=None):
        return build_session_metadata(
            session, find_episodes(session), size, face, arm, DEFAULT_TASK, DEFAULT_SORTING_RULES
        )

    def test_one_entry_per_episode_with_explicit_values(self, session):
        rec = self._build(session)
        assert [e["episode_dir"] for e in rec["episodes"]] == ["0001", "0002", "0003"]
        for entry in rec["episodes"]:
            assert entry["envelope_size"] == "big"
            assert entry["envelope_facing_side"] == "upside"

    def test_destination_basket_is_derived_from_sorting_rules(self, session):
        assert self._build(session, size="small")["envelope"]["destination_basket_side"] == "right"
        assert self._build(session, size="big")["envelope"]["destination_basket_side"] == "left"
        assert self._build(session, size="medium")["envelope"]["destination_basket_side"] == "left"

    def test_aborted_episodes_are_reported(self, session):
        rec = self._build(session)
        assert rec["session"]["aborted_count"] == 1
        assert rec["session"]["aborted_episodes"] == ["0002"]

    def test_recorder_status_is_carried_through(self, session):
        rec = self._build(session)
        by_dir = {e["episode_dir"]: e for e in rec["episodes"]}
        assert by_dir["0002"]["recorder_status"] == "aborted"
        assert by_dir["0001"]["recorder_status"] == "success"

    def test_status_is_recorded_not_planned(self, session):
        """The label describes what exists, not what was intended."""
        assert self._build(session)["metadata_status"] == "recorded"

    def test_arm_omitted_when_not_given(self, session):
        assert "arm" not in self._build(session)["envelope"]

    def test_arm_included_when_given(self, session):
        assert self._build(session, arm="left")["envelope"]["arm"] == "left"

    def test_episode_count_matches(self, session):
        assert self._build(session)["session"]["episode_count"] == 3


# =============================================================================
# stamping episodes
# =============================================================================


class TestStampEpisode:
    ENVELOPE = {
        "envelope_size": "big",
        "envelope_facing_side": "upside",
        "destination_basket_side": "left",
    }

    def test_preserves_existing_recorder_fields(self, session):
        stamp_episode(session / "0002", self.ENVELOPE)
        data = read_sidecar(session / "0002")
        assert data["status"] == "aborted"      # the recorder's verdict is not ours to rewrite
        assert data["version"] == 1
        assert data["duration"] == 4
        assert data["envelope_size"] == "big"

    def test_creates_sidecar_when_missing(self, tmp_path):
        d = tmp_path / "0001"
        d.mkdir()
        stamp_episode(d, self.ENVELOPE)
        assert read_sidecar(d)["envelope_size"] == "big"

    def test_is_idempotent(self, session):
        stamp_episode(session / "0001", self.ENVELOPE)
        first = read_sidecar(session / "0001")
        stamp_episode(session / "0001", self.ENVELOPE)
        assert read_sidecar(session / "0001") == first

    def test_output_is_valid_json(self, session):
        stamp_episode(session / "0001", self.ENVELOPE)
        json.loads((session / "0001" / "metadata.json").read_text())


# =============================================================================
# conflict detection
# =============================================================================


class TestFindConflicts:
    ENVELOPE = {
        "envelope_size": "big",
        "envelope_facing_side": "upside",
        "destination_basket_side": "left",
    }

    def test_unlabelled_session_has_no_conflicts(self, session):
        assert find_conflicts(find_episodes(session), self.ENVELOPE) == []

    def test_same_label_again_is_not_a_conflict(self, session):
        for d in find_episodes(session):
            stamp_episode(d, self.ENVELOPE)
        assert find_conflicts(find_episodes(session), self.ENVELOPE) == []

    def test_different_label_is_flagged_per_episode(self, session):
        for d in find_episodes(session):
            stamp_episode(d, self.ENVELOPE)
        other = {**self.ENVELOPE, "envelope_size": "small"}
        conflicts = find_conflicts(find_episodes(session), other)
        assert len(conflicts) == 3
        assert "envelope_size='big'" in conflicts[0]

    def test_conflict_names_the_episode(self, session):
        stamp_episode(session / "0001", self.ENVELOPE)
        other = {**self.ENVELOPE, "envelope_facing_side": "downside"}
        assert find_conflicts([session / "0001"], other)[0].startswith("0001:")
