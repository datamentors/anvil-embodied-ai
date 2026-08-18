#!/usr/bin/env python3
"""Create and validate an isolated, train-ready LeRobot v3 dataset copy.

This utility repairs statistics without ever modifying the source dataset:

* copies the source into a hidden sibling staging directory (reflink when
  supported, regular copy otherwise; hard links are explicitly rejected),
* recomputes every global and per-episode statistic from the actual Parquet
  rows and decoded video frames,
* writes only ``meta/stats.json`` and the ``stats/*`` columns embedded in
  ``meta/episodes/**/*.parquet`` inside staging,
* proves that data, videos, tasks, info and all other immutable artifacts are
  byte-for-byte identical to the source,
* validates LeRobot loading and Pi0.5's quantile-normalization contract, and
* publishes the result with a no-replace atomic directory rename only after a
  ``TRAIN_READY.json`` marker has been written and fsynced.

The repair deliberately does not call LeRobot 0.5.1's ``aggregate_stats``.
That function averages per-episode quantiles, which is not the quantile of the
full dataset.  Here q01/q10/q50/q90/q99 are computed once over the full value
population using NumPy's linear-quantile definition.  Video quantiles use an
equivalent exact histogram calculation over all decoded uint8 RGB pixels.

Typical use for an absolute-action Pi0.5 dataset with a 10-frame
action-from-observation lookahead (do not run from a different LeRobot
environment):

    /path/to/anvil/.venv/bin/python prepare_trainready_dataset.py \
      /path/to/reviewed-dataset --afo-lookahead-frames=10

The default output is the source's sibling named with ``-trainready``.

Validation-only mode performs no writes:

    /path/to/anvil/.venv/bin/python prepare_trainready_dataset.py \
      /path/to/reviewed-dataset-trainready \
      --validate-only
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EXPECTED_LEROBOT_VERSION = "0.5.1"
QUANTILES: tuple[tuple[str, float], ...] = (
    ("q01", 0.01),
    ("q10", 0.10),
    ("q50", 0.50),
    ("q90", 0.90),
    ("q99", 0.99),
)
STAT_NAMES: tuple[str, ...] = (
    "min",
    "max",
    "mean",
    "std",
    "count",
    *(name for name, _ in QUANTILES),
)
MARKER_NAME = "TRAIN_READY.json"
MUTABLE_GLOBAL_STATS = Path("meta/stats.json")
MUTABLE_EPISODE_PREFIX = "meta/episodes/"
ALGORITHM_ID = "full-population-linear-quantiles-v1"


class PreparationError(RuntimeError):
    """Raised when any safety or data-integrity invariant is violated."""


@dataclass(frozen=True)
class EpisodePart:
    relative_path: Path
    row_count: int


@dataclass
class CameraStatsResult:
    key: str
    global_stats: dict[str, np.ndarray]
    episode_stats: list[dict[str, np.ndarray]]
    decoded_frames: int
    decoded_files: int
    width: int
    height: int


@dataclass
class ExpectedMetadata:
    info: dict[str, Any]
    data_table: pa.Table
    episode_table: pa.Table
    episode_parts: list[EpisodePart]
    global_stats: dict[str, dict[str, np.ndarray]]
    episode_stats: dict[str, list[dict[str, np.ndarray]]]
    camera_results: dict[str, CameraStatsResult]


@dataclass(frozen=True)
class LoaderFacts:
    action_chunk_size: int
    task_prompts: tuple[str, ...]


def _say(message: str) -> None:
    print(message, flush=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            manifest[path.relative_to(root).as_posix()] = _sha256_file(path)
    return manifest


def _manifest_digest(manifest: dict[str, str]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_repaired_metadata(relative: str) -> bool:
    return relative == MUTABLE_GLOBAL_STATS.as_posix() or relative.startswith(
        MUTABLE_EPISODE_PREFIX
    )


def _immutable_manifest(manifest: dict[str, str]) -> dict[str, str]:
    return {
        relative: digest
        for relative, digest in manifest.items()
        if not _is_repaired_metadata(relative) and relative != MARKER_NAME
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=4, sort_keys=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_parquet(path: Path, table: pa.Table) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        pq.write_table(table, temporary, compression="snappy")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_rename_noreplace(source: Path, target: Path) -> None:
    """Atomically rename a directory without replacing an existing target."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is not None:
        at_fdcwd = -100
        rename_noreplace = 1
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(target),
            rename_noreplace,
        )
        if result == 0:
            _fsync_directory(target.parent)
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise PreparationError(f"target appeared during publish: {target}")
        if error not in (errno.ENOSYS, errno.EINVAL):
            raise OSError(error, os.strerror(error), target)

    # Fallback for non-Linux systems.  The production workstation supports
    # renameat2; this branch exists to keep local validation portable.
    if target.exists():
        raise PreparationError(f"target already exists: {target}")
    os.rename(source, target)
    _fsync_directory(target.parent)


def _lerobot_version() -> str:
    try:
        return importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PreparationError(
            "LeRobot is not installed. Run this with the Anvil snapshot's .venv Python."
        ) from exc


def _validate_root(root: Path) -> dict[str, Any]:
    _require(root.is_absolute(), f"dataset path must be absolute: {root}")
    _require(root.is_dir(), f"dataset directory not found: {root}")
    info_path = root / "meta/info.json"
    _require(info_path.is_file(), f"missing {info_path}")
    with info_path.open(encoding="utf-8") as stream:
        info = json.load(stream)
    _require(info.get("codebase_version") == "v3.0", "dataset is not LeRobot v3.0")
    _require(int(info.get("total_episodes", 0)) > 0, "dataset has no episodes")
    _require(int(info.get("total_frames", 0)) > 0, "dataset has no frames")
    _require(int(info.get("fps", 0)) > 0, "dataset fps must be positive")
    return info


def _load_data_table(root: Path) -> pa.Table:
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    _require(bool(files), f"no data parquet files below {root / 'data'}")
    tables = [pq.read_table(path) for path in files]
    return pa.concat_tables(tables, promote_options="default")


def _load_episode_table(root: Path) -> tuple[pa.Table, list[EpisodePart]]:
    files = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    _require(bool(files), f"no episode metadata below {root / 'meta/episodes'}")
    tables: list[pa.Table] = []
    parts: list[EpisodePart] = []
    for path in files:
        table = pq.read_table(path)
        tables.append(table)
        parts.append(EpisodePart(path.relative_to(root), table.num_rows))
    return pa.concat_tables(tables, promote_options="default"), parts


def _column_numpy(table: pa.Table, key: str) -> np.ndarray:
    _require(key in table.column_names, f"data parquet is missing feature {key!r}")
    values = np.asarray(table[key].to_pylist())
    _require(values.shape[0] == table.num_rows, f"invalid row count for {key}")
    return values


def _vector_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """Exact full-population stats with LeRobot-compatible output shapes."""
    array = np.asarray(values)
    _require(array.shape[0] >= 1, "cannot compute statistics for an empty array")
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    array64 = array.astype(np.float64, copy=False)
    result: dict[str, np.ndarray] = {
        "min": np.min(array64, axis=0),
        "max": np.max(array64, axis=0),
        "mean": np.mean(array64, axis=0, dtype=np.float64),
        "std": np.std(array64, axis=0, dtype=np.float64),
        "count": np.asarray([array64.shape[0]], dtype=np.int64),
    }
    for name, quantile in QUANTILES:
        result[name] = np.quantile(array64, quantile, axis=0, method="linear")
    return result


def _histogram_quantile(histogram: np.ndarray, quantile: float) -> np.ndarray:
    """NumPy-linear quantile from exact uint8 per-channel histograms."""
    _require(histogram.ndim == 2 and histogram.shape[1] == 256, "invalid RGB histogram")
    values: list[float] = []
    for channel_histogram in histogram:
        count = int(channel_histogram.sum(dtype=np.uint64))
        _require(count > 0, "empty camera histogram")
        rank = (count - 1) * quantile
        lower_rank = math.floor(rank)
        upper_rank = math.ceil(rank)
        fraction = rank - lower_rank
        cumulative = np.cumsum(channel_histogram, dtype=np.uint64)
        lower_value = int(np.searchsorted(cumulative, lower_rank, side="right"))
        upper_value = int(np.searchsorted(cumulative, upper_rank, side="right"))
        values.append(((1.0 - fraction) * lower_value + fraction * upper_value) / 255.0)
    return np.asarray(values, dtype=np.float64)


def _histogram_stats(histogram: np.ndarray, frame_count: int) -> dict[str, np.ndarray]:
    """Exact decoded-RGB stats in LeRobot's (3,1,1) camera shape."""
    _require(frame_count > 0, "camera feature has no frames")
    histogram = np.asarray(histogram, dtype=np.uint64)
    channel_counts = histogram.sum(axis=1, dtype=np.uint64)
    _require(np.all(channel_counts == channel_counts[0]), "RGB channel counts differ")

    levels = np.arange(256, dtype=np.float64)
    count = float(channel_counts[0])
    means_u8 = (histogram * levels[None, :]).sum(axis=1, dtype=np.float64) / count
    seconds_u8 = (histogram * np.square(levels)[None, :]).sum(axis=1, dtype=np.float64) / count
    variance_u8 = np.maximum(0.0, seconds_u8 - np.square(means_u8))

    minimum = (
        np.asarray([np.flatnonzero(channel)[0] for channel in histogram], dtype=np.float64) / 255.0
    )
    maximum = (
        np.asarray([np.flatnonzero(channel)[-1] for channel in histogram], dtype=np.float64) / 255.0
    )
    result: dict[str, np.ndarray] = {
        "min": minimum.reshape(3, 1, 1),
        "max": maximum.reshape(3, 1, 1),
        "mean": (means_u8 / 255.0).reshape(3, 1, 1),
        "std": (np.sqrt(variance_u8) / 255.0).reshape(3, 1, 1),
        # LeRobot get_feature_stats uses the number of images/frames as count,
        # while reducing pixel values over axes (0,2,3).
        "count": np.asarray([frame_count], dtype=np.int64),
    }
    for name, quantile in QUANTILES:
        result[name] = _histogram_quantile(histogram, quantile).reshape(3, 1, 1)
    return result


def _video_path(
    root: Path,
    info: dict[str, Any],
    key: str,
    chunk_index: int,
    file_index: int,
) -> Path:
    template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    return root / template.format(
        video_key=key,
        chunk_index=chunk_index,
        file_index=file_index,
    )


def _decode_camera_stats(
    root: Path,
    info: dict[str, Any],
    episode_table: pa.Table,
    key: str,
) -> CameraStatsResult:
    fps = int(info["fps"])
    shape = tuple(info["features"][key]["shape"])
    _require(len(shape) == 3 and shape[0] == 3, f"unsupported camera shape for {key}: {shape}")
    expected_height, expected_width = int(shape[1]), int(shape[2])
    episode_rows = episode_table.to_pylist()
    episode_count = len(episode_rows)

    segments_by_file: dict[Path, list[tuple[int, int, int]]] = {}
    for row_number, row in enumerate(episode_rows):
        chunk = int(row[f"videos/{key}/chunk_index"])
        file_index = int(row[f"videos/{key}/file_index"])
        start_seconds = float(row[f"videos/{key}/from_timestamp"])
        start_float = start_seconds * fps
        start = round(start_float)
        _require(
            abs(start_float - start) <= 1e-3,
            f"{key} episode {row_number} starts off the {fps} Hz grid: {start_seconds}",
        )
        length = int(row["length"])
        end = start + length
        end_seconds = float(row[f"videos/{key}/to_timestamp"])
        _require(
            abs(end_seconds * fps - end) <= 1e-3,
            f"{key} episode {row_number} video duration disagrees with length",
        )
        path = _video_path(root, info, key, chunk, file_index)
        _require(path.is_file(), f"missing video: {path}")
        segments_by_file.setdefault(path, []).append((start, end, row_number))

    episode_histograms = np.zeros((episode_count, 3, 256), dtype=np.uint64)
    episode_frame_counts = np.zeros(episode_count, dtype=np.int64)
    decoded_frames = 0

    def flush_batch(frames: list[np.ndarray], episode_numbers: list[int]) -> None:
        if not frames:
            return
        batch = np.stack(frames, axis=0)
        episode_array = np.asarray(episode_numbers, dtype=np.int64)
        for episode_number in np.unique(episode_array):
            selected = batch[episode_array == episode_number]
            for channel in range(3):
                counts = np.bincount(selected[..., channel].reshape(-1), minlength=256)
                episode_histograms[episode_number, channel] += counts.astype(np.uint64)
            episode_frame_counts[episode_number] += selected.shape[0]

    for path, segments in sorted(segments_by_file.items(), key=lambda item: str(item[0])):
        expected_file_frames = max(end for _, end, _ in segments)
        assignments = np.full(expected_file_frames, -1, dtype=np.int64)
        for start, end, episode_number in segments:
            _require(
                np.all(assignments[start:end] == -1),
                f"overlapping video metadata in {path}",
            )
            assignments[start:end] = episode_number
        _require(np.all(assignments >= 0), f"gaps in video metadata for {path}")

        batch_frames: list[np.ndarray] = []
        batch_episodes: list[int] = []
        file_frames = 0
        with av.open(str(path), mode="r") as container:
            video_streams = container.streams.video
            _require(len(video_streams) == 1, f"expected one video stream in {path}")
            for frame in container.decode(video=0):
                _require(
                    file_frames < len(assignments),
                    f"{path} contains more frames than metadata declares",
                )
                rgb = frame.to_ndarray(format="rgb24")
                _require(
                    rgb.shape == (expected_height, expected_width, 3),
                    f"unexpected decoded shape in {path}: {rgb.shape}",
                )
                batch_frames.append(rgb)
                batch_episodes.append(int(assignments[file_frames]))
                file_frames += 1
                if len(batch_frames) >= 16:
                    flush_batch(batch_frames, batch_episodes)
                    batch_frames.clear()
                    batch_episodes.clear()
            flush_batch(batch_frames, batch_episodes)

        _require(
            file_frames == expected_file_frames,
            f"{path}: decoded {file_frames} frames, expected {expected_file_frames}",
        )
        decoded_frames += file_frames

    expected_episode_lengths = np.asarray(
        [int(row["length"]) for row in episode_rows], dtype=np.int64
    )
    _require(
        np.array_equal(episode_frame_counts, expected_episode_lengths),
        f"decoded per-episode frame counts do not match metadata for {key}",
    )
    global_histogram = episode_histograms.sum(axis=0, dtype=np.uint64)
    global_stats = _histogram_stats(global_histogram, decoded_frames)
    per_episode = [
        _histogram_stats(episode_histograms[index], int(episode_frame_counts[index]))
        for index in range(episode_count)
    ]
    return CameraStatsResult(
        key=key,
        global_stats=global_stats,
        episode_stats=per_episode,
        decoded_frames=decoded_frames,
        decoded_files=len(segments_by_file),
        width=expected_width,
        height=expected_height,
    )


def _validate_row_contracts(
    info: dict[str, Any], data_table: pa.Table, episode_table: pa.Table
) -> None:
    total_frames = int(info["total_frames"])
    total_episodes = int(info["total_episodes"])
    _require(data_table.num_rows == total_frames, "info.total_frames disagrees with Parquet")
    _require(
        episode_table.num_rows == total_episodes, "info.total_episodes disagrees with metadata"
    )

    data_episode = _column_numpy(data_table, "episode_index").astype(np.int64)
    data_index = _column_numpy(data_table, "index").astype(np.int64)
    _require(
        np.array_equal(data_index, np.arange(total_frames, dtype=np.int64)),
        "data index must be exactly 0..total_frames-1",
    )

    rows = episode_table.to_pylist()
    episode_indices = [int(row["episode_index"]) for row in rows]
    _require(
        episode_indices == list(range(total_episodes)),
        "episode metadata must be ordered and indexed 0..N-1",
    )
    cursor = 0
    for row in rows:
        episode_index = int(row["episode_index"])
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        length = int(row["length"])
        _require(start == cursor, f"episode {episode_index} starts at {start}, expected {cursor}")
        _require(end - start == length, f"episode {episode_index} has inconsistent length")
        _require(
            np.all(data_episode[start:end] == episode_index),
            f"episode_index data mismatch in episode {episode_index}",
        )
        frame_index = _column_numpy(data_table.slice(start, length), "frame_index").astype(np.int64)
        _require(
            np.array_equal(frame_index, np.arange(length, dtype=np.int64)),
            f"frame_index is not contiguous in episode {episode_index}",
        )
        cursor = end
    _require(cursor == total_frames, "episode ranges do not cover all frames")


def _compute_expected_metadata(root: Path) -> ExpectedMetadata:
    info = _validate_root(root)
    data_table = _load_data_table(root)
    episode_table, episode_parts = _load_episode_table(root)
    _validate_row_contracts(info, data_table, episode_table)

    features: dict[str, dict[str, Any]] = info["features"]
    numeric_keys = [
        key
        for key, feature in features.items()
        if feature.get("dtype") not in {"video", "image", "string"}
    ]
    camera_keys = [
        key for key, feature in features.items() if feature.get("dtype") in {"video", "image"}
    ]
    _require(camera_keys, "dataset has no camera features")

    global_stats: dict[str, dict[str, np.ndarray]] = {}
    episode_stats: dict[str, list[dict[str, np.ndarray]]] = {}
    episode_rows = episode_table.to_pylist()

    for key in numeric_keys:
        values = _column_numpy(data_table, key)
        _require(np.all(np.isfinite(values)), f"{key} contains NaN or infinity")
        global_stats[key] = _vector_stats(values)
        per_episode: list[dict[str, np.ndarray]] = []
        for row in episode_rows:
            start = int(row["dataset_from_index"])
            end = int(row["dataset_to_index"])
            per_episode.append(_vector_stats(values[start:end]))
        episode_stats[key] = per_episode

    camera_results: dict[str, CameraStatsResult] = {}
    max_workers = min(len(camera_keys), 3)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_decode_camera_stats, root, info, episode_table, key): key
            for key in camera_keys
        }
        for future in as_completed(futures):
            key = futures[future]
            result = future.result()
            camera_results[key] = result
            global_stats[key] = result.global_stats
            episode_stats[key] = result.episode_stats
            _say(f"decoded {key}: {result.decoded_frames} frames")

    # Preserve feature order from info.json.  This is not semantically required,
    # but keeps stats.json deterministic and easy to inspect.
    global_stats = {key: global_stats[key] for key in features if key in global_stats}
    episode_stats = {key: episode_stats[key] for key in features if key in episode_stats}

    return ExpectedMetadata(
        info=info,
        data_table=data_table,
        episode_table=episode_table,
        episode_parts=episode_parts,
        global_stats=global_stats,
        episode_stats=episode_stats,
        camera_results=camera_results,
    )


def _jsonable_stats(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    return {
        feature: {name: np.asarray(value).tolist() for name, value in feature_stats.items()}
        for feature, feature_stats in stats.items()
    }


def _replace_episode_stats_columns(
    table: pa.Table,
    episode_stats: dict[str, list[dict[str, np.ndarray]]],
) -> pa.Table:
    result = table
    for feature, rows in episode_stats.items():
        _require(len(rows) == table.num_rows, f"wrong episode stats row count for {feature}")
        for stat_name in STAT_NAMES:
            column_name = f"stats/{feature}/{stat_name}"
            column_index = result.schema.get_field_index(column_name)
            _require(column_index >= 0, f"episode metadata is missing {column_name}")
            field = result.schema.field(column_index)
            values = [np.asarray(row[stat_name]).tolist() for row in rows]
            array = pa.array(values, type=field.type)
            result = result.set_column(column_index, field, array)
    return result


def _write_repaired_metadata(root: Path, expected: ExpectedMetadata) -> None:
    _atomic_write_json(root / MUTABLE_GLOBAL_STATS, _jsonable_stats(expected.global_stats))

    repaired_table = _replace_episode_stats_columns(expected.episode_table, expected.episode_stats)
    cursor = 0
    for part in expected.episode_parts:
        path = root / part.relative_path
        _atomic_write_parquet(path, repaired_table.slice(cursor, part.row_count))
        cursor += part.row_count
    _require(cursor == repaired_table.num_rows, "episode metadata partitioning error")


def _assert_stats_equal(
    actual: dict[str, Any], expected: dict[str, dict[str, np.ndarray]], context: str
) -> None:
    _require(set(actual) == set(expected), f"{context}: feature keys differ")
    for feature, feature_expected in expected.items():
        feature_actual = actual[feature]
        _require(
            set(feature_actual) == set(feature_expected),
            f"{context}/{feature}: statistic keys differ",
        )
        for stat_name, expected_value in feature_expected.items():
            actual_value = np.asarray(feature_actual[stat_name])
            expected_array = np.asarray(expected_value)
            _require(
                actual_value.shape == expected_array.shape,
                f"{context}/{feature}/{stat_name}: shape {actual_value.shape} != {expected_array.shape}",
            )
            _require(
                np.array_equal(actual_value, expected_array),
                f"{context}/{feature}/{stat_name}: values differ; "
                f"max_abs_error={np.max(np.abs(actual_value - expected_array))}",
            )


def _validate_episode_stats(table: pa.Table, expected: ExpectedMetadata) -> None:
    for feature, rows in expected.episode_stats.items():
        for stat_name in STAT_NAMES:
            column_name = f"stats/{feature}/{stat_name}"
            _require(column_name in table.column_names, f"missing {column_name}")
            actual_rows = table[column_name].to_pylist()
            for episode_number, (actual, expected_row) in enumerate(
                zip(actual_rows, rows, strict=True)
            ):
                actual_array = np.asarray(actual)
                expected_array = np.asarray(expected_row[stat_name])
                _require(
                    np.array_equal(actual_array, expected_array),
                    f"episode {episode_number}/{feature}/{stat_name} differs from source rows",
                )


def _validate_non_stats_episode_columns(source: Path, candidate: Path) -> None:
    source_table, _ = _load_episode_table(source)
    candidate_table, _ = _load_episode_table(candidate)
    source_columns = [name for name in source_table.column_names if not name.startswith("stats/")]
    candidate_columns = [
        name for name in candidate_table.column_names if not name.startswith("stats/")
    ]
    _require(source_columns == candidate_columns, "non-stats episode columns changed")
    _require(
        source_table.select(source_columns).equals(candidate_table.select(candidate_columns)),
        "non-stats episode metadata is not Arrow-equal to the source",
    )


def _validate_quantile_normalization(expected: ExpectedMetadata) -> None:
    """Validate the exact normalization contract used by Pi0.5."""
    for feature in ("observation.state", "action"):
        _require(feature in expected.global_stats, f"missing Pi0.5 feature {feature}")
        values = _column_numpy(expected.data_table, feature).astype(np.float64, copy=False)
        stats = expected.global_stats[feature]
        q01 = np.asarray(stats["q01"], dtype=np.float64)
        q99 = np.asarray(stats["q99"], dtype=np.float64)
        denominator = q99 - q01
        _require(np.all(denominator >= 0), f"{feature} has inverted q01/q99")
        # This is the exact zero-range behavior of LeRobot 0.5.1's
        # NormalizerProcessorStep: constant dimensions use eps as denominator.
        nondegenerate = denominator > 0
        safe_denominator = np.where(nondegenerate, denominator, 1e-8)
        normalized = 2.0 * (values - q01) / safe_denominator - 1.0
        _require(np.all(np.isfinite(normalized)), f"{feature} normalization is non-finite")
        normalized_q01 = np.quantile(normalized, 0.01, axis=0, method="linear")
        normalized_q99 = np.quantile(normalized, 0.99, axis=0, method="linear")
        _require(
            np.allclose(normalized_q01[nondegenerate], -1.0, rtol=0.0, atol=1e-12),
            f"{feature} q01 does not map to -1",
        )
        _require(
            np.allclose(normalized_q99[nondegenerate], 1.0, rtol=0.0, atol=1e-12),
            f"{feature} q99 does not map to +1",
        )
        restored = (normalized + 1.0) * safe_denominator / 2.0 + q01
        _require(
            np.allclose(restored, values, rtol=0.0, atol=1e-12),
            f"{feature} normalization round-trip failed",
        )


def _validate_lerobot_loading(root: Path, expected: ExpectedMetadata) -> LoaderFacts:
    """Exercise the exact local-root loading path used by anvil-trainer."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    metadata = LeRobotDatasetMetadata(repo_id="local", root=root)
    policy_config = PI05Config(device="cpu")
    delta_timestamps = resolve_delta_timestamps(policy_config, metadata)
    dataset = LeRobotDataset(
        repo_id="local",
        root=root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    _require(
        dataset.num_episodes == int(expected.info["total_episodes"]),
        "LeRobot episode count differs",
    )
    _require(
        dataset.num_frames == int(expected.info["total_frames"]), "LeRobot frame count differs"
    )
    _require(dataset.fps == int(expected.info["fps"]), "LeRobot fps differs")

    camera_keys = sorted(expected.camera_results)
    sample_indices = sorted(
        {
            0,
            dataset.num_frames // 2,
            dataset.num_frames - 1,
            *(int(row["dataset_from_index"]) for row in expected.episode_table.to_pylist()[::10]),
        }
    )
    state_shape = tuple(expected.info["features"]["observation.state"]["shape"])
    action_shape = tuple(expected.info["features"]["action"]["shape"])
    camera_shapes = {key: tuple(expected.info["features"][key]["shape"]) for key in camera_keys}
    action_chunk_size: int | None = None
    task_prompts: set[str] = set()
    for index in sample_indices:
        item = dataset[index]
        _require(
            tuple(item["observation.state"].shape) == state_shape,
            f"unexpected state shape at frame {index}",
        )
        item_action_shape = tuple(item["action"].shape)
        _require(
            len(item_action_shape) == len(action_shape) + 1
            and item_action_shape[1:] == action_shape,
            f"unexpected Pi0.5 action chunk shape at frame {index}: {item_action_shape}",
        )
        if action_chunk_size is None:
            action_chunk_size = item_action_shape[0]
        _require(
            item_action_shape[0] == action_chunk_size,
            f"Pi0.5 action chunk size changed at frame {index}",
        )
        prompt = item.get("task")
        _require(
            isinstance(prompt, str) and bool(prompt.strip()),
            f"missing task prompt at frame {index}",
        )
        task_prompts.add(prompt)
        for key in camera_keys:
            _require(
                tuple(item[key].shape) == camera_shapes[key],
                f"unexpected {key} shape at frame {index}",
            )

    _require(action_chunk_size is not None, "Pi0.5 loader returned no samples")
    return LoaderFacts(
        action_chunk_size=action_chunk_size,
        task_prompts=tuple(sorted(task_prompts)),
    )


def _validate_candidate(
    root: Path,
    *,
    source: Path | None,
    source_immutable_manifest: dict[str, str] | None,
    require_marker: bool,
) -> tuple[ExpectedMetadata, dict[str, Any]]:
    expected = _compute_expected_metadata(root)

    with (root / MUTABLE_GLOBAL_STATS).open(encoding="utf-8") as stream:
        actual_global = json.load(stream)
    _assert_stats_equal(actual_global, expected.global_stats, "meta/stats.json")

    actual_episode_table, _ = _load_episode_table(root)
    _validate_episode_stats(actual_episode_table, expected)
    _validate_quantile_normalization(expected)
    loader_facts = _validate_lerobot_loading(root, expected)

    candidate_manifest = _file_manifest(root)
    if source is not None:
        _require(source_immutable_manifest is not None, "missing source manifest")
        candidate_immutable = _immutable_manifest(candidate_manifest)
        _require(
            candidate_immutable == source_immutable_manifest,
            "an immutable file differs from the source",
        )
        _validate_non_stats_episode_columns(source, root)

    marker: dict[str, Any] | None = None
    marker_path = root / MARKER_NAME
    if require_marker:
        _require(marker_path.is_file(), f"missing {marker_path}")
        with marker_path.open(encoding="utf-8") as stream:
            marker = json.load(stream)
        _require(marker.get("status") == "train-ready", "TRAIN_READY marker status is invalid")
        _require(marker.get("algorithm") == ALGORITHM_ID, "TRAIN_READY algorithm is unexpected")
        _require(
            marker.get("facts", {}).get("frames") == int(expected.info["total_frames"]),
            "TRAIN_READY frame count differs",
        )

    facts = {
        "episodes": int(expected.info["total_episodes"]),
        "frames": int(expected.info["total_frames"]),
        "fps": int(expected.info["fps"]),
        "camera_keys": sorted(expected.camera_results),
        "joint_names": expected.info["features"]["action"].get("names"),
        "task_prompts": list(loader_facts.task_prompts),
        "pi05_chunk_size": loader_facts.action_chunk_size,
        "manifest_sha256": _manifest_digest(candidate_manifest),
        "marker_present": marker is not None,
    }
    return expected, facts


def _copy_to_staging(source: Path, target: Path) -> Path:
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=str(source.parent)))
    _say(f"staging: {staging}")
    try:
        subprocess.run(
            ["cp", "-a", "--reflink=auto", f"{source}/.", str(staging)],
            check=True,
        )
        shutil.copystat(source, staging, follow_symlinks=False)
    except Exception:
        _say(f"copy failed; staging retained for inspection: {staging}")
        raise

    # A reflink has a distinct inode.  Reject hard links to ensure a metadata
    # write in staging can never mutate the source inode.
    for relative in _file_manifest(source):
        source_path = source / relative
        staging_path = staging / relative
        source_stat = source_path.stat()
        staging_stat = staging_path.stat()
        _require(
            (source_stat.st_dev, source_stat.st_ino) != (staging_stat.st_dev, staging_stat.st_ino),
            f"hard-linked copy rejected: {relative}",
        )
    return staging


def _default_target(source: Path) -> Path:
    if source.name.endswith("-trainready"):
        raise PreparationError("source already ends in -trainready; pass --validate-only")
    return source.with_name(f"{source.name}-trainready")


def _validate_paths(
    source: Path, target: Path | None, validate_only: bool
) -> tuple[Path, Path | None]:
    source = source.expanduser().resolve()
    _validate_root(source)
    if validate_only:
        _require(target is None, "--target cannot be combined with --validate-only")
        return source, None

    target = (target or _default_target(source)).expanduser()
    target = (Path.cwd() / target).resolve() if not target.is_absolute() else target.resolve()
    _require(target.parent == source.parent, "target must be a sibling of the source dataset")
    _require(target != source, "target and source must differ")
    _require(not target.exists(), f"target already exists: {target}")
    return source, target


def _build_marker(
    *,
    source: Path,
    target: Path,
    source_manifest: dict[str, str],
    facts: dict[str, Any],
    expected: ExpectedMetadata,
    action_type: str,
    afo_lookahead_frames: int,
) -> dict[str, Any]:
    return {
        "status": "train-ready",
        "algorithm": ALGORITHM_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "source_dataset": str(source),
        "dataset": str(target),
        "lerobot_version": _lerobot_version(),
        "numpy_version": np.__version__,
        "pyarrow_version": pa.__version__,
        "pyav_version": av.__version__,
        "source_manifest_sha256": _manifest_digest(source_manifest),
        "immutable_manifest": _immutable_manifest(source_manifest),
        "facts": {
            **facts,
            "action_type": action_type,
            "afo_lookahead_frames": afo_lookahead_frames,
            "camera_decode": {
                key: {
                    "frames": result.decoded_frames,
                    "files": result.decoded_files,
                    "width": result.width,
                    "height": result.height,
                }
                for key, result in sorted(expected.camera_results.items())
            },
        },
        "validation": {
            "global_stats_recomputed": True,
            "per_episode_stats_recomputed": True,
            "all_video_frames_decoded": True,
            "immutable_files_sha256_equal": True,
            "episode_non_stats_arrow_equal": True,
            "pi05_quantile_normalization_checked": True,
            "lerobot_local_loader_checked": True,
        },
    }


def prepare(
    source: Path,
    target: Path,
    *,
    action_type: str = "absolute",
    afo_lookahead_frames: int = 0,
) -> dict[str, Any]:
    version = _lerobot_version()
    _require(
        version == EXPECTED_LEROBOT_VERSION,
        f"expected LeRobot {EXPECTED_LEROBOT_VERSION}, found {version}",
    )

    source_manifest_before = _file_manifest(source)
    source_immutable = _immutable_manifest(source_manifest_before)
    staging = _copy_to_staging(source, target)
    try:
        staging_manifest_before = _file_manifest(staging)
        _require(
            staging_manifest_before == source_manifest_before,
            "staging copy is not byte-for-byte identical to source",
        )

        _say("computing exact full-population statistics (first pass)")
        expected = _compute_expected_metadata(staging)
        _write_repaired_metadata(staging, expected)

        _say("validating repaired dataset independently (second full pass)")
        validated_expected, facts = _validate_candidate(
            staging,
            source=source,
            source_immutable_manifest=source_immutable,
            require_marker=False,
        )

        source_manifest_after = _file_manifest(source)
        _require(
            source_manifest_after == source_manifest_before,
            "source dataset changed during preparation",
        )

        marker = _build_marker(
            source=source,
            target=target,
            source_manifest=source_manifest_before,
            facts=facts,
            expected=validated_expected,
            action_type=action_type,
            afo_lookahead_frames=afo_lookahead_frames,
        )
        _atomic_write_json(staging / MARKER_NAME, marker)
        _fsync_directory(staging)

        _require(not target.exists(), f"target appeared before publish: {target}")
        _atomic_rename_noreplace(staging, target)
        _say(f"published train-ready dataset: {target}")
        return marker
    except Exception:
        _say(f"preparation failed; source is untouched and staging is retained: {staging}")
        raise


def validate_only(root: Path) -> dict[str, Any]:
    version = _lerobot_version()
    _require(
        version == EXPECTED_LEROBOT_VERSION,
        f"expected LeRobot {EXPECTED_LEROBOT_VERSION}, found {version}",
    )
    _, facts = _validate_candidate(
        root,
        source=None,
        source_immutable_manifest=None,
        require_marker=True,
    )
    _say(f"validation passed: {root}")
    return facts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="reviewed dataset, or train-ready dataset in validate-only mode",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="output sibling (default: SOURCE-trainready)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="perform complete read-only validation of an existing train-ready dataset",
    )
    parser.add_argument(
        "--action-type",
        choices=("absolute", "delta_obs_t", "delta_sequential"),
        default="absolute",
        help="action representation recorded in TRAIN_READY.json (default: absolute)",
    )
    parser.add_argument(
        "--afo-lookahead-frames",
        type=int,
        default=0,
        help="action-from-observation lookahead recorded in TRAIN_READY.json",
    )
    args = parser.parse_args(argv)
    if args.afo_lookahead_frames < 0:
        parser.error("--afo-lookahead-frames must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source, target = _validate_paths(args.source, args.target, args.validate_only)
        if args.validate_only:
            result = validate_only(source)
        else:
            _require(target is not None, "internal target resolution error")
            result = prepare(
                source,
                target,
                action_type=args.action_type,
                afo_lookahead_frames=args.afo_lookahead_frames,
            )
        print(json.dumps(result, indent=2, sort_keys=False))
        return 0
    except (
        PreparationError,
        OSError,
        ValueError,
        pa.ArrowException,
        av.error.FFmpegError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
