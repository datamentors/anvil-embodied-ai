"""Tests for carrying recorder labels into the converted dataset's episode metadata."""
from __future__ import annotations

import json

import pytest

from mcap_converter.core.episode_labels import (
    ENVELOPE_KEYS,
    MISSING,
    SOURCE_KEYS,
    build_label_table,
    install_episode_metadata_injector,
    read_episode_labels,
    read_sidecar,
    resolve_label_keys,
    source_columns,
)


def make_episode(root, name, sidecar: dict | None):
    """Create one episode directory with an mcap and (optionally) a sidecar."""
    d = root / name
    d.mkdir(parents=True)
    mcap = d / f"{name}_0.mcap"
    mcap.write_bytes(b"stub")
    if sidecar is not None:
        (d / "metadata.json").write_text(json.dumps(sidecar))
    return mcap


LABELLED = {
    "version": 1,
    "status": "success",
    "envelope_size": "big",
    "envelope_facing_side": "upside",
    "destination_basket_side": "left",
}


@pytest.fixture
def session(tmp_path):
    """Three labelled episodes."""
    return [make_episode(tmp_path / "sess", n, dict(LABELLED)) for n in ("0001", "0002", "0003")]


# =============================================================================
# reading
# =============================================================================


class TestReadSidecar:
    def test_reads_json(self, tmp_path):
        m = make_episode(tmp_path / "s", "0001", LABELLED)
        assert read_sidecar(m)["envelope_size"] == "big"

    def test_missing_sidecar_is_empty(self, tmp_path):
        m = make_episode(tmp_path / "s", "0001", None)
        assert read_sidecar(m) == {}

    def test_corrupt_sidecar_is_empty(self, tmp_path):
        m = make_episode(tmp_path / "s", "0001", None)
        (m.parent / "metadata.json").write_text("{ not json")
        assert read_sidecar(m) == {}


class TestSourceColumns:
    def test_records_dir_and_file(self, tmp_path):
        m = make_episode(tmp_path / "s", "0007", LABELLED)
        assert source_columns(m) == {"source_episode_dir": "0007", "source_mcap": "0007_0.mcap"}


class TestReadEpisodeLabels:
    def test_returns_requested_keys_as_strings(self, tmp_path):
        m = make_episode(tmp_path / "s", "0001", {**LABELLED, "arm": "left"})
        got = read_episode_labels(m, ("envelope_size", "arm", "source_episode_dir"))
        assert got == {"envelope_size": "big", "arm": "left", "source_episode_dir": "0001"}

    def test_omits_keys_the_sidecar_lacks(self, tmp_path):
        m = make_episode(tmp_path / "s", "0001", {"version": 1})
        assert read_episode_labels(m, ("envelope_size",)) == {}

    def test_ignores_keys_not_asked_for(self, tmp_path):
        m = make_episode(tmp_path / "s", "0001", LABELLED)
        assert "status" not in read_episode_labels(m, ENVELOPE_KEYS)


# =============================================================================
# key-set resolution
# =============================================================================


class TestResolveLabelKeys:
    def test_source_keys_are_always_present(self, session, tmp_path):
        keys = resolve_label_keys(session, tmp_path / "out")
        assert set(SOURCE_KEYS) <= set(keys)

    def test_includes_labels_that_exist(self, session, tmp_path):
        keys = resolve_label_keys(session, tmp_path / "out")
        assert "envelope_size" in keys and "envelope_facing_side" in keys

    def test_omits_labels_no_episode_has(self, session, tmp_path):
        """An unlabelled session must not gain empty columns."""
        assert "arm" not in resolve_label_keys(session, tmp_path / "out")

    def test_unlabelled_session_gets_only_source_keys(self, tmp_path):
        eps = [make_episode(tmp_path / "s", n, {"version": 1}) for n in ("0001", "0002")]
        assert set(resolve_label_keys(eps, tmp_path / "out")) == set(SOURCE_KEYS)

    def test_one_labelled_episode_is_enough(self, tmp_path):
        eps = [
            make_episode(tmp_path / "s", "0001", {"version": 1}),
            make_episode(tmp_path / "s", "0002", LABELLED),
        ]
        assert "envelope_size" in resolve_label_keys(eps, tmp_path / "out")

    def test_extra_keys_are_honoured(self, tmp_path):
        eps = [make_episode(tmp_path / "s", "0001", {"operator": "ana"})]
        keys = resolve_label_keys(eps, tmp_path / "out", extra_keys=("operator",))
        assert "operator" in keys

    def test_resume_reuses_existing_columns(self, session, tmp_path):
        """Appending to an existing parquet must not change the schema."""
        import pandas as pd

        out = tmp_path / "out"
        ep_dir = out / "meta" / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True)
        pd.DataFrame(
            {"episode_index": [0], "source_episode_dir": ["0001"], "source_mcap": ["a.mcap"]}
        ).to_parquet(ep_dir / "file-000.parquet")

        keys = resolve_label_keys(session, out, resume_from=1)
        assert set(keys) == set(SOURCE_KEYS)   # envelope columns absent before: stay absent

    def test_resume_on_dataset_without_meta_adds_nothing(self, session, tmp_path):
        assert resolve_label_keys(session, tmp_path / "empty", resume_from=1) == ()


# =============================================================================
# label table
# =============================================================================


class TestBuildLabelTable:
    def test_every_episode_gets_the_same_key_set(self, tmp_path):
        """The pyarrow flush fails when key sets differ between episodes."""
        eps = [
            make_episode(tmp_path / "s", "0001", LABELLED),
            make_episode(tmp_path / "s", "0002", {"version": 1}),   # unlabelled
        ]
        keys = resolve_label_keys(eps, tmp_path / "out")
        table, _ = build_label_table(eps, keys)
        assert all(set(row) == set(keys) for row in table.values())

    def test_missing_values_become_empty_strings(self, tmp_path):
        eps = [
            make_episode(tmp_path / "s", "0001", LABELLED),
            make_episode(tmp_path / "s", "0002", {"version": 1}),
        ]
        keys = resolve_label_keys(eps, tmp_path / "out")
        table, _ = build_label_table(eps, keys)
        assert table[str(eps[1])]["envelope_size"] == MISSING

    def test_incomplete_episodes_are_reported(self, tmp_path):
        eps = [
            make_episode(tmp_path / "s", "0001", LABELLED),
            make_episode(tmp_path / "s", "0002", {"version": 1}),
        ]
        keys = resolve_label_keys(eps, tmp_path / "out")
        _, incomplete = build_label_table(eps, keys)
        assert incomplete == ["0002"]

    def test_fully_labelled_session_reports_nothing_missing(self, session, tmp_path):
        keys = resolve_label_keys(session, tmp_path / "out")
        _, incomplete = build_label_table(session, keys)
        assert incomplete == []

    def test_source_columns_are_filled_per_episode(self, session, tmp_path):
        keys = resolve_label_keys(session, tmp_path / "out")
        table, _ = build_label_table(session, keys)
        assert [table[str(p)]["source_episode_dir"] for p in session] == ["0001", "0002", "0003"]


# =============================================================================
# injector
# =============================================================================


class _FakeMeta:
    def __init__(self):
        self.calls = []

    def save_episode(self, *args):
        # Mirrors LeRobotDatasetMetadata.save_episode; only the last argument
        # (episode_metadata) is what the injector merges into.
        self.calls.append(dict(args[-1]))


class _FakeDataset:
    def __init__(self):
        self.meta = _FakeMeta()


class TestInstallEpisodeMetadataInjector:
    def test_extra_keys_reach_save_episode(self):
        ds = _FakeDataset()
        extra = install_episode_metadata_injector(ds)
        extra.update({"envelope_size": "big"})
        ds.meta.save_episode(0, 10, ["t"], {}, {"length": 10})
        assert ds.meta.calls[0] == {"length": 10, "envelope_size": "big"}

    def test_original_metadata_is_preserved(self):
        ds = _FakeDataset()
        install_episode_metadata_injector(ds)
        ds.meta.save_episode(0, 10, ["t"], {}, {"data/chunk_index": 3})
        assert ds.meta.calls[0]["data/chunk_index"] == 3

    def test_each_episode_gets_the_current_contents(self):
        ds = _FakeDataset()
        extra = install_episode_metadata_injector(ds)
        for i, size in enumerate(("big", "small")):
            extra.clear()
            extra.update({"envelope_size": size})
            ds.meta.save_episode(i, 10, ["t"], {}, {})
        assert [c["envelope_size"] for c in ds.meta.calls] == ["big", "small"]

    def test_empty_extra_changes_nothing(self):
        ds = _FakeDataset()
        install_episode_metadata_injector(ds)
        ds.meta.save_episode(0, 10, ["t"], {}, {"length": 10})
        assert ds.meta.calls[0] == {"length": 10}
