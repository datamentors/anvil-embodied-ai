"""Tests for the session-valid CLI (a converted session checked against its raw input)."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mcap_converter.cli.session_valid import main

CAMERAS = ("base", "left_wrist", "right_wrist")
LABELS = {"envelope_size": "big", "envelope_facing_side": "upside"}
CHECK_NAMES = [
    "episode_accounting",
    "info_json",
    "episode_lengths",
    "data_rows",
    "action_state",
    "frames_vs_duration",
    "videos",
    "labels",
    "provenance",
]


def write_raw(root, episodes, *, severity=None, duration=None, labels=None):
    """Raw session: one folder per episode with a stub MCAP and metadata.json, plus the report."""
    raw = root / "raw"
    entries = []
    for ep in episodes:
        folder = raw / ep
        folder.mkdir(parents=True)
        mcap = folder / f"{ep}_0.mcap"
        mcap.write_bytes(b"stub")
        (folder / "metadata.json").write_text(json.dumps({"status": "success", **(labels or {})}))
        entries.append(
            {
                "path": str(mcap.resolve()),
                "severity": (severity or {}).get(ep, "pass"),
                "duration_s": (duration or {}).get(ep, 2.0),
                "topics": [],
            }
        )
    report = raw / "mcap_valid_reports" / "report.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"episodes": entries}))
    return raw


def write_dataset(
    root,
    converted,
    *,
    fps=30,
    length=60,
    equal=None,
    labels=None,
    source_columns=True,
    cameras=CAMERAS,
    provenance=True,
    codebase_version="v3.0",
):
    """A LeRobot v3-shaped tree: info.json, meta/episodes, data, videos, conversion_config.yaml."""
    ds = root / "dataset"
    lengths = {ep: (length.get(ep, 60) if isinstance(length, dict) else length) for ep in converted}
    features = {
        f"observation.images.{c}": {"dtype": "video", "shape": [3, 48, 64]} for c in cameras
    }
    features["observation.state"] = {"dtype": "float32", "shape": [4]}
    features["action"] = {"dtype": "float32", "shape": [4]}
    info = {
        "codebase_version": codebase_version,
        "fps": fps,
        "total_episodes": len(converted),
        "total_frames": sum(lengths.values()),
        "features": features,
    }
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text(json.dumps(info))

    rows = {"episode_index": list(range(len(converted))), "length": list(lengths.values())}
    if source_columns:
        rows["source_episode_dir"] = list(converted)
        rows["source_mcap"] = [f"{ep}_0.mcap" for ep in converted]
    for key, value in (labels or {}).items():
        rows[key] = [value] * len(converted)
    (ds / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table(rows), ds / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    episode_index, actions, states = [], [], []
    rng = np.random.default_rng(0)
    for i, ep in enumerate(converted):
        n = lengths[ep]
        state = rng.normal(size=(n, 4)).astype(np.float32)
        action = state + 0.1
        same = int(round(n * (equal or {}).get(ep, 0.0)))
        action[:same] = state[:same]
        episode_index += [i] * n
        actions += action.tolist()
        states += state.tolist()
    vector = pa.list_(pa.float32(), 4)
    data = pa.table(
        {
            "episode_index": pa.array(episode_index, pa.int64()),
            "action": pa.array(actions, vector),
            "observation.state": pa.array(states, vector),
        }
    )
    (ds / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(data, ds / "data" / "chunk-000" / "file-000.parquet")

    for camera in cameras:
        video = ds / "videos" / f"observation.images.{camera}" / "chunk-000" / "file-000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"not really an mp4")

    config = "robot_state_topic: /joint_states\n"
    if provenance:
        config += "\n# --- provenance ---\ncode_commit: b300789\n"
    (ds / "conversion_config.yaml").write_text(config)
    return ds


def write_exclusions(root, total, kept, excluded=()):
    path = root / "exclusions.json"
    path.write_text(
        json.dumps(
            {
                "excluded": [{"episode": e, "reasons": ["manual"]} for e in excluded],
                "kept": kept,
                "total": total,
            }
        )
    )
    return path


def run(root, *extra):
    out = root / "work" / "session_valid.json"
    code = main(
        [
            "--raw",
            str(root / "raw"),
            "--dataset",
            str(root / "dataset"),
            "--exclusions",
            str(root / "exclusions.json"),
            "--out",
            str(out),
            *extra,
        ]
    )
    result = json.loads(out.read_text())
    return code, result, {c["name"]: c for c in result["checks"]}


@pytest.fixture
def clean(tmp_path):
    """Three raw episodes, one excluded beforehand, two converted with labels."""
    write_raw(tmp_path, ["0001", "0003"], labels=LABELS)
    write_dataset(tmp_path, ["0001", "0003"], labels=LABELS)
    write_exclusions(tmp_path, total=3, kept=2, excluded=["0002"])
    return tmp_path


def test_clean_session_passes(clean):
    code, result, checks = run(clean)
    assert code == 0
    assert result["status"] == "pass"
    assert set(result) == {"status", "checks", "episodes"}
    assert [c["name"] for c in result["checks"]] == CHECK_NAMES
    assert all(set(c) == {"name", "status", "detail"} for c in result["checks"])
    assert {c["status"] for c in result["checks"]} == {"pass"}
    assert result["episodes"] == {"raw": 3, "dataset": 2, "excluded": 1, "skipped": 0}


def test_critical_episode_is_accounted_as_skipped(tmp_path):
    write_raw(tmp_path, ["0001", "0002", "0003"], severity={"0002": "critical"}, labels=LABELS)
    write_dataset(tmp_path, ["0001", "0003"], labels=LABELS)
    write_exclusions(tmp_path, total=3, kept=3)
    code, result, checks = run(tmp_path)
    assert code == 0
    assert checks["episode_accounting"]["status"] == "pass"
    assert result["episodes"] == {"raw": 3, "dataset": 2, "excluded": 0, "skipped": 1}


def test_include_flagged_pass_also_skips_warnings(tmp_path):
    write_raw(tmp_path, ["0001", "0002"], severity={"0002": "warning"}, labels=LABELS)
    write_dataset(tmp_path, ["0001"], labels=LABELS)
    write_exclusions(tmp_path, total=2, kept=2)
    assert run(tmp_path)[2]["episode_accounting"]["status"] == "warn"
    code, result, checks = run(tmp_path, "--include-flagged", "pass")
    assert checks["episode_accounting"]["status"] == "pass"
    assert result["episodes"]["skipped"] == 1


def test_episode_dropped_by_the_converter_warns(tmp_path):
    write_raw(tmp_path, ["0001", "0002", "0003"], labels=LABELS)
    write_dataset(tmp_path, ["0001", "0003"], labels=LABELS)
    write_exclusions(tmp_path, total=3, kept=3)
    code, result, checks = run(tmp_path)
    assert code == 0
    assert result["status"] == "warn"
    assert "0002" in checks["episode_accounting"]["detail"]


def test_converted_critical_episode_fails(tmp_path):
    write_raw(tmp_path, ["0001", "0002"], severity={"0002": "critical"}, labels=LABELS)
    write_dataset(tmp_path, ["0001", "0002"], labels=LABELS)
    write_exclusions(tmp_path, total=2, kept=2)
    code, _, checks = run(tmp_path)
    assert code == 1
    assert checks["episode_accounting"]["status"] == "fail"


def test_raw_that_changed_since_exclusion_fails(clean):
    write_exclusions(clean, total=3, kept=1, excluded=["0002", "0004"])
    code, _, checks = run(clean)
    assert code == 1
    assert "exclusions kept 1" in checks["episode_accounting"]["detail"]


def test_missing_exclusions_fails(clean):
    (clean / "exclusions.json").unlink()
    code, _, checks = run(clean)
    assert code == 1
    assert checks["episode_accounting"]["status"] == "fail"


def test_without_source_columns_counts_must_match(tmp_path):
    write_raw(tmp_path, ["0001", "0002", "0003"])
    write_dataset(tmp_path, ["0001", "0002", "0003"], source_columns=False)
    write_exclusions(tmp_path, total=3, kept=3)
    assert run(tmp_path)[2]["episode_accounting"]["status"] == "pass"

    (tmp_path / "raw" / "0003" / "0003_0.mcap").unlink()
    write_exclusions(tmp_path, total=2, kept=2)
    code, _, checks = run(tmp_path)
    assert code == 1
    assert "expected 2" in checks["episode_accounting"]["detail"]


def test_info_json_version_and_fps(clean):
    info_path = clean / "dataset" / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    assert run(clean, "--fps", "30")[2]["info_json"]["status"] == "pass"
    assert run(clean, "--fps", "60")[2]["info_json"]["status"] == "fail"
    info["codebase_version"] = "v2.1"
    info_path.write_text(json.dumps(info))
    code, _, checks = run(clean)
    assert code == 1
    assert "v2.1" in checks["info_json"]["detail"]


def test_missing_info_json_fails(clean):
    (clean / "dataset" / "meta" / "info.json").unlink()
    code, _, checks = run(clean)
    assert code == 1
    assert checks["info_json"]["status"] == "fail"


def test_lengths_must_add_up_to_total_frames(clean):
    info_path = clean / "dataset" / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] += 5
    info_path.write_text(json.dumps(info))
    _, _, checks = run(clean)
    assert checks["episode_lengths"]["status"] == "fail"
    assert checks["data_rows"]["status"] == "fail"


@pytest.mark.parametrize(
    "fraction, flags, expected",
    [
        (0.1, [], "pass"),
        (0.3, [], "warn"),
        (0.6, [], "fail"),
        (0.3, ["--action-state-warn", "0.4"], "pass"),
        (0.6, ["--action-state-fail", "0.7"], "warn"),
    ],
)
def test_action_equal_to_state(tmp_path, fraction, flags, expected):
    write_raw(tmp_path, ["0001", "0002"], labels=LABELS)
    write_dataset(tmp_path, ["0001", "0002"], labels=LABELS, equal={"0002": fraction})
    write_exclusions(tmp_path, total=2, kept=2)
    code, _, checks = run(tmp_path, *flags)
    assert checks["action_state"]["status"] == expected
    assert code == (1 if expected == "fail" else 0)
    if expected != "pass":
        assert "1 (0002)" in checks["action_state"]["detail"]


def test_frames_far_from_recorded_duration_warn(tmp_path):
    write_raw(tmp_path, ["0001", "0002"], duration={"0002": 3.0}, labels=LABELS)
    write_dataset(tmp_path, ["0001", "0002"], labels=LABELS)
    write_exclusions(tmp_path, total=2, kept=2)
    code, _, checks = run(tmp_path)
    assert code == 0
    assert checks["frames_vs_duration"]["status"] == "warn"
    assert "0002" in checks["frames_vs_duration"]["detail"]
    assert run(tmp_path, "--frames-ratio-min", "0.6")[2]["frames_vs_duration"]["status"] == "pass"


def test_expected_cameras(clean):
    assert run(clean, "--expected-cameras", "3")[2]["videos"]["status"] == "pass"
    code, _, checks = run(clean, "--expected-cameras", "2")
    assert code == 1
    assert checks["videos"]["status"] == "fail"


def test_empty_video_fails(clean):
    video = clean / "dataset" / "videos" / "observation.images.base" / "chunk-000" / "file-000.mp4"
    video.write_bytes(b"")
    code, _, checks = run(clean)
    assert code == 1
    assert "observation.images.base" in checks["videos"]["detail"]


def test_missing_labels_warn_unless_required(tmp_path):
    write_raw(tmp_path, ["0001"])
    write_dataset(tmp_path, ["0001"])
    write_exclusions(tmp_path, total=1, kept=1)
    code, result, checks = run(tmp_path)
    assert (code, result["status"], checks["labels"]["status"]) == (0, "warn", "warn")
    code, _, checks = run(tmp_path, "--require-labels")
    assert (code, checks["labels"]["status"]) == (1, "fail")


def test_partly_empty_labels(tmp_path):
    write_raw(tmp_path, ["0001"], labels=LABELS)
    write_dataset(tmp_path, ["0001"], labels={**LABELS, "envelope_facing_side": ""})
    write_exclusions(tmp_path, total=1, kept=1)
    _, _, checks = run(tmp_path)
    assert checks["labels"]["status"] == "fail"
    assert "envelope_facing_side" in checks["labels"]["detail"]


def test_labels_that_differ_from_raw_fail(tmp_path):
    write_raw(tmp_path, ["0001"], labels={**LABELS, "envelope_size": "small"})
    write_dataset(tmp_path, ["0001"], labels=LABELS)
    write_exclusions(tmp_path, total=1, kept=1)
    code, _, checks = run(tmp_path)
    assert code == 1
    assert "envelope_size='big' vs raw 'small'" in checks["labels"]["detail"]


def test_custom_label_keys(tmp_path):
    write_raw(tmp_path, ["0001"], labels={"towel_color": "blue"})
    write_dataset(tmp_path, ["0001"], labels={"towel_color": "blue"})
    write_exclusions(tmp_path, total=1, kept=1)
    _, _, checks = run(tmp_path, "--require-labels", "--labels", "towel_color")
    assert checks["labels"]["status"] == "pass"
    _, _, checks = run(tmp_path, "--require-labels", "--labels", "towel_color,fold_type")
    assert checks["labels"]["status"] == "fail"


def test_provenance(clean):
    config = clean / "dataset" / "conversion_config.yaml"
    assert "code_commit=b300789" in run(clean)[2]["provenance"]["detail"]
    config.write_text("robot_state_topic: /joint_states\n")
    assert run(clean)[2]["provenance"]["status"] == "warn"
    config.unlink()
    code, _, checks = run(clean)
    assert (code, checks["provenance"]["status"]) == (1, "fail")


def test_missing_dataset_still_writes_a_result(tmp_path):
    write_raw(tmp_path, ["0001"])
    write_exclusions(tmp_path, total=1, kept=1)
    code, result, checks = run(tmp_path)
    assert code == 1
    assert result["status"] == "fail"
    assert [c["name"] for c in result["checks"]] == CHECK_NAMES
