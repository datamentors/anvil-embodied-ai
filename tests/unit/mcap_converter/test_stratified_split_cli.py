"""Tests for the stratified-split CLI's label parsing and normalisation."""
from __future__ import annotations

import json

import pytest

from mcap_converter.cli.stratified_split import (
    _FACE_ALIASES,
    _SIZE_ALIASES,
    FACE_COLUMNS,
    SIZE_COLUMNS,
    _norm,
    check_coverage,
    load_labels_from_file,
    load_labels_from_meta,
)

# =============================================================================
# value normalisation
# =============================================================================


class TestNormalisation:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("upside", "face_up"),
            ("downside", "face_down"),
            ("UPSIDE", "face_up"),
            ("  Downside  ", "face_down"),
            ("face up", "face_up"),      # space -> underscore
            ("up", "face_up"),
            ("down", "face_down"),
            ("face_up", "face_up"),
            ("face-down", "face_down"),
            ("cima", "face_up"),
            ("baixo", "face_down"),
        ],
    )
    def test_face_aliases(self, raw, expected):
        assert _norm(raw, _FACE_ALIASES, "face") == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("big", "big"), ("BIG", "big"), ("large", "big"), ("grande", "big"),
            ("medium", "medium"), ("med", "medium"), ("M", "medium"),
            ("small", "small"), ("pequeno", "small"),
        ],
    )
    def test_size_aliases(self, raw, expected):
        assert _norm(raw, _SIZE_ALIASES, "size") == expected

    def test_unknown_value_is_rejected_with_the_accepted_list(self):
        with pytest.raises(ValueError, match="unrecognised face value"):
            _norm("sideways", _FACE_ALIASES, "face")

    def test_error_names_the_episode_when_known(self):
        with pytest.raises(ValueError, match="for episode 7"):
            _norm("huge", _SIZE_ALIASES, "size", ep=7)

    def test_upside_downside_are_registered_both_ways(self):
        """Guards the alignment with the recording-protocol vocabulary."""
        assert _FACE_ALIASES["upside"] == "face_up"
        assert _FACE_ALIASES["downside"] == "face_down"


# =============================================================================
# label files
# =============================================================================


class TestLoadLabelsFromFile:
    def test_grouped_json_with_upside_downside(self, tmp_path):
        p = tmp_path / "labels.json"
        p.write_text(json.dumps({
            "small": {"upside": [0, 1], "downside": [2]},
            "big": {"upside": [3], "downside": [4]},
        }))
        assert load_labels_from_file(p) == {
            0: "small|face_up", 1: "small|face_up", 2: "small|face_down",
            3: "big|face_up", 4: "big|face_down",
        }

    def test_flat_json_accepts_facing_side_key(self, tmp_path):
        p = tmp_path / "labels.json"
        p.write_text(json.dumps({
            "0": {"envelope_size": "big", "envelope_facing_side": "upside"},
            "1": {"size": "small", "face": "downside"},
        }))
        assert load_labels_from_file(p) == {0: "big|face_up", 1: "small|face_down"}

    def test_flat_json_list_and_string_forms(self, tmp_path):
        p = tmp_path / "labels.json"
        p.write_text(json.dumps({"0": ["medium", "upside"], "1": "small|downside"}))
        assert load_labels_from_file(p) == {0: "medium|face_up", 1: "small|face_down"}

    def test_csv_with_facing_side_column(self, tmp_path):
        p = tmp_path / "labels.csv"
        p.write_text("episode_index,envelope_size,envelope_facing_side\n0,big,upside\n1,small,downside\n")
        assert load_labels_from_file(p) == {0: "big|face_up", 1: "small|face_down"}

    def test_conflicting_labels_are_rejected(self, tmp_path):
        p = tmp_path / "labels.json"
        p.write_text(json.dumps({
            "big": {"upside": [0]},
            "small": {"downside": [0]},   # same episode, two strata
        }))
        with pytest.raises(ValueError, match="labelled both"):
            load_labels_from_file(p)

    def test_repeating_the_same_label_is_not_a_conflict(self, tmp_path):
        p = tmp_path / "labels.json"
        p.write_text(json.dumps({"big": {"upside": [0, 0]}}))
        assert load_labels_from_file(p) == {0: "big|face_up"}

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_labels_from_file(tmp_path / "nope.json")

    def test_empty_json_object(self, tmp_path):
        p = tmp_path / "labels.json"
        p.write_text("{}")
        with pytest.raises(ValueError, match="no labels"):
            load_labels_from_file(p)


# =============================================================================
# episode metadata columns
# =============================================================================


class TestLoadLabelsFromMeta:
    def _dataset(self, tmp_path, size_col, face_col, sizes, faces):
        import pandas as pd

        root = tmp_path / "ds"
        (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": len(sizes)}))
        pd.DataFrame({
            "episode_index": range(len(sizes)),
            size_col: sizes,
            face_col: faces,
        }).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        return root

    def test_reads_envelope_facing_side_column(self, tmp_path):
        root = self._dataset(
            tmp_path, "envelope_size", "envelope_facing_side",
            ["big", "small"], ["upside", "downside"],
        )
        assert load_labels_from_meta(root, None, None) == {
            0: "big|face_up", 1: "small|face_down",
        }

    def test_explicit_column_names_win(self, tmp_path):
        root = self._dataset(
            tmp_path, "envelope_size", "orientation", ["medium"], ["upside"],
        )
        assert load_labels_from_meta(root, "envelope_size", "orientation") == {
            0: "medium|face_up"
        }

    def test_missing_column_points_at_the_labels_flag(self, tmp_path):
        import pandas as pd

        root = tmp_path / "ds"
        (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        pd.DataFrame({"episode_index": [0]}).to_parquet(
            root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )
        with pytest.raises(ValueError, match="--labels"):
            load_labels_from_meta(root, None, None)

    def test_unknown_explicit_column_is_rejected(self, tmp_path):
        root = self._dataset(tmp_path, "envelope_size", "envelope_face", ["big"], ["upside"])
        with pytest.raises(ValueError, match="--size-column"):
            load_labels_from_meta(root, "nope", None)

    def test_facing_side_is_in_the_column_candidates(self):
        assert "envelope_facing_side" in FACE_COLUMNS
        assert "envelope_size" in SIZE_COLUMNS


# =============================================================================
# coverage checks
# =============================================================================


class TestCheckCoverage:
    def test_full_coverage_passes(self):
        check_coverage({0: "a", 1: "a"}, 2, allow_partial=False)

    def test_missing_labels_are_rejected_by_default(self):
        with pytest.raises(ValueError, match="have no label"):
            check_coverage({0: "a"}, 3, allow_partial=False)

    def test_allow_partial_permits_gaps(self, capsys):
        check_coverage({0: "a"}, 3, allow_partial=True)
        assert "WARNING" in capsys.readouterr().out

    def test_episode_beyond_dataset_is_always_rejected(self):
        """Guards against labels written against pre-merge episode numbering."""
        with pytest.raises(ValueError, match="outside the dataset"):
            check_coverage({0: "a", 99: "a"}, 2, allow_partial=True)


class TestEmptyLabelsFromConverter:
    """mcap-convert writes "" for episodes the recorder never labelled."""

    def _dataset(self, tmp_path, sizes, faces):
        import pandas as pd

        root = tmp_path / "ds"
        (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": len(sizes)}))
        pd.DataFrame({
            "episode_index": range(len(sizes)),
            "envelope_size": sizes,
            "envelope_facing_side": faces,
        }).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        return root

    def test_blank_values_are_treated_as_unlabelled(self, tmp_path):
        root = self._dataset(tmp_path, ["big", "", "small"], ["upside", "", "downside"])
        assert load_labels_from_meta(root, None, None) == {0: "big|face_up", 2: "small|face_down"}

    def test_blank_then_reported_by_coverage_not_as_a_bad_value(self, tmp_path):
        root = self._dataset(tmp_path, ["big", ""], ["upside", ""])
        labels = load_labels_from_meta(root, None, None)
        with pytest.raises(ValueError, match="have no label"):
            check_coverage(labels, 2, allow_partial=False)
