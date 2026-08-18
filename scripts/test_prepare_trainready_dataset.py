from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

SCRIPT_PATH = Path(__file__).with_name("prepare_trainready_dataset.py")
SPEC = importlib.util.spec_from_file_location("prepare_trainready_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_trainready_dataset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_trainready_dataset
SPEC.loader.exec_module(prepare_trainready_dataset)


def test_vector_stats_use_full_population_linear_quantiles() -> None:
    values = np.asarray([[0.0, 10.0], [1.0, 20.0], [100.0, 30.0]])

    stats = prepare_trainready_dataset._vector_stats(values)

    assert np.array_equal(stats["count"], np.asarray([3]))
    for name, quantile in prepare_trainready_dataset.QUANTILES:
        assert np.array_equal(
            stats[name],
            np.quantile(values, quantile, axis=0, method="linear"),
        )


def test_histogram_quantiles_match_numpy_linear_quantiles() -> None:
    channels = [
        np.asarray([0, 0, 255], dtype=np.uint8),
        np.asarray([10, 20, 30], dtype=np.uint8),
        np.asarray([4, 4, 8], dtype=np.uint8),
    ]
    histogram = np.stack(
        [np.bincount(channel, minlength=256) for channel in channels],
        axis=0,
    )

    for _, quantile in prepare_trainready_dataset.QUANTILES:
        actual = prepare_trainready_dataset._histogram_quantile(histogram, quantile)
        expected = np.asarray(
            [np.quantile(channel, quantile, method="linear") / 255.0 for channel in channels]
        )
        assert np.allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_cli_records_dataset_contract_without_experiment_hardcoding() -> None:
    args = prepare_trainready_dataset.parse_args(
        [
            "/dataset",
            "--action-type=delta_obs_t",
            "--afo-lookahead-frames=7",
        ]
    )

    assert args.source == Path("/dataset")
    assert args.action_type == "delta_obs_t"
    assert args.afo_lookahead_frames == 7


def test_cli_rejects_negative_lookahead() -> None:
    with pytest.raises(SystemExit):
        prepare_trainready_dataset.parse_args(["/dataset", "--afo-lookahead-frames=-1"])


def test_marker_uses_inferred_loader_facts_and_explicit_action_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    expected = prepare_trainready_dataset.ExpectedMetadata(
        info={"total_episodes": 2, "total_frames": 20, "fps": 30},
        data_table=pa.table({"index": pa.array([], type=pa.int64())}),
        episode_table=pa.table({"episode_index": pa.array([], type=pa.int64())}),
        episode_parts=[],
        global_stats={},
        episode_stats={},
        camera_results={},
    )
    facts = {
        "episodes": 2,
        "frames": 20,
        "fps": 30,
        "camera_keys": ["observation.images.base"],
        "joint_names": ["joint1"],
        "task_prompts": ["A dataset-specific prompt"],
        "pi05_chunk_size": 50,
        "manifest_sha256": "abc",
        "marker_present": False,
    }

    marker = prepare_trainready_dataset._build_marker(
        source=source,
        target=target,
        source_manifest={},
        facts=facts,
        expected=expected,
        action_type="absolute",
        afo_lookahead_frames=10,
    )

    assert marker["facts"]["task_prompts"] == ["A dataset-specific prompt"]
    assert marker["facts"]["pi05_chunk_size"] == 50
    assert marker["facts"]["action_type"] == "absolute"
    assert marker["facts"]["afo_lookahead_frames"] == 10
    json.dumps(marker)


def test_default_target_is_sibling_and_never_reuses_trainready_source() -> None:
    source = Path("/datasets/example")
    assert prepare_trainready_dataset._default_target(source) == Path(
        "/datasets/example-trainready"
    )
    with pytest.raises(prepare_trainready_dataset.PreparationError):
        prepare_trainready_dataset._default_target(Path("/datasets/example-trainready"))
