"""GPU-path runtime patches for faster LeRobot conversion."""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

_PATCH_APPLIED = False
_ORIGINAL_COMPUTE_EPISODE_STATS = None

_DEFAULT_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
_DEFAULT_SAMPLE_LIMIT = 16_384

logger = logging.getLogger(__name__)


def _quantile_keys(quantile_list: tuple[float, ...]) -> list[str]:
    return [f"q{int(q * 100):02d}" for q in quantile_list]


def _compute_vector_stats(
    array: np.ndarray,
    *,
    quantile_list: tuple[float, ...] = _DEFAULT_QUANTILES,
) -> dict[str, np.ndarray]:
    """Compute per-column stats for episode vectors without histogram bookkeeping."""
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    stats: dict[str, np.ndarray] = {
        "min": np.min(arr, axis=0).astype(np.float32, copy=False),
        "max": np.max(arr, axis=0).astype(np.float32, copy=False),
        "mean": np.mean(arr, axis=0, dtype=np.float64).astype(np.float32, copy=False),
        "std": np.std(arr, axis=0, dtype=np.float64).astype(np.float32, copy=False),
        "count": np.array([arr.shape[0]], dtype=np.int64),
    }
    for key, q in zip(_quantile_keys(quantile_list), quantile_list, strict=True):
        stats[key] = np.quantile(arr, q, axis=0).astype(np.float32, copy=False)

    if array.ndim == 1:
        for key, value in list(stats.items()):
            if key == "count":
                continue
            stats[key] = np.atleast_1d(value.reshape(1))

    return stats


class _FastVideoStats:
    """Cheap per-channel running stats for video frames."""

    def __init__(
        self,
        *,
        quantile_list: tuple[float, ...] = _DEFAULT_QUANTILES,
        sample_limit: int = _DEFAULT_SAMPLE_LIMIT,
    ) -> None:
        self.quantile_list = quantile_list
        self.sample_limit = sample_limit
        self.count = 0
        self.sum = np.zeros(3, dtype=np.float64)
        self.sum_sq = np.zeros(3, dtype=np.float64)
        self.min = np.full(3, 255.0, dtype=np.float64)
        self.max = np.zeros(3, dtype=np.float64)
        self.samples: list[np.ndarray] = []
        self.sample_count = 0

    def update(self, frame_hwc: np.ndarray) -> None:
        if frame_hwc.ndim != 3 or frame_hwc.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB frame, got shape {frame_hwc.shape}")

        height, width, _ = frame_hwc.shape
        stride = max(1, max(height, width) // 160)
        sampled = frame_hwc[::stride, ::stride].reshape(-1, 3).astype(np.float32, copy=False)
        if sampled.size == 0:
            return

        self.count += sampled.shape[0]
        self.sum += sampled.sum(axis=0, dtype=np.float64)
        self.sum_sq += np.square(sampled, dtype=np.float64).sum(axis=0, dtype=np.float64)
        self.min = np.minimum(self.min, sampled.min(axis=0))
        self.max = np.maximum(self.max, sampled.max(axis=0))

        if self.sample_count < self.sample_limit:
            remaining = self.sample_limit - self.sample_count
            if sampled.shape[0] > remaining:
                step = max(1, sampled.shape[0] // remaining)
                sampled = sampled[::step][:remaining]
            self.samples.append(sampled)
            self.sample_count += sampled.shape[0]

    def get_statistics(self) -> dict[str, np.ndarray] | None:
        if self.count < 2:
            return None

        mean = self.sum / self.count
        variance = np.maximum(0.0, self.sum_sq / self.count - np.square(mean))
        sample_bank = np.concatenate(self.samples, axis=0) if self.samples else np.zeros((0, 3))

        stats: dict[str, np.ndarray] = {
            "min": self.min.astype(np.float32),
            "max": self.max.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": np.sqrt(variance).astype(np.float32),
            "count": np.array([self.count], dtype=np.int64),
        }
        for key, q in zip(_quantile_keys(self.quantile_list), self.quantile_list, strict=True):
            if sample_bank.size == 0:
                stats[key] = mean.astype(np.float32)
            else:
                stats[key] = np.quantile(sample_bank, q, axis=0).astype(np.float32, copy=False)
        return stats


def compute_fast_episode_stats(
    episode_data: dict[str, list[str] | np.ndarray],
    features: dict,
    quantile_list: list[float] | None = None,
) -> dict:
    """Fast episode stats for GPU conversion.

    Numeric arrays are handled directly with NumPy. Image/video features fall back
    to LeRobot's implementation if they appear outside the streaming-encoder path.
    """
    if quantile_list is None:
        q_tuple = _DEFAULT_QUANTILES
    else:
        q_tuple = tuple(quantile_list)

    ep_stats = {}
    for key, data in episode_data.items():
        dtype = features[key]["dtype"]
        if dtype == "string":
            continue
        if dtype in {"image", "video"}:
            if _ORIGINAL_COMPUTE_EPISODE_STATS is None:
                raise RuntimeError("Original compute_episode_stats is unavailable")
            fallback_stats = _ORIGINAL_COMPUTE_EPISODE_STATS({key: data}, {key: features[key]}, list(q_tuple))
            ep_stats.update(fallback_stats)
            continue
        ep_stats[key] = _compute_vector_stats(np.asarray(data), quantile_list=q_tuple)
    return ep_stats


def _fast_feed_frame(self, video_key: str, image: np.ndarray) -> None:
    """Queue frames without an unconditional deep copy."""
    if not self._episode_active:
        raise RuntimeError("No active episode. Call start_episode() first.")

    thread = self._threads[video_key]
    if not thread.is_alive():
        try:
            status, msg = self._result_queues[video_key].get_nowait()
            if status == "error":
                raise RuntimeError(f"Encoder thread for {video_key} crashed: {msg}")
        except queue.Empty:
            pass
        raise RuntimeError(f"Encoder thread for {video_key} is not alive")

    frame = image
    if not isinstance(frame, np.ndarray):
        frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = frame.transpose(1, 2, 0)
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    elif not frame.flags.c_contiguous:
        frame = np.ascontiguousarray(frame)

    try:
        self._frame_queues[video_key].put(frame, timeout=0.1)
    except queue.Full:
        self._dropped_frames[video_key] = self._dropped_frames.get(video_key, 0) + 1
        count = self._dropped_frames[video_key]
        if count == 1 or count % 10 == 0:
            logger.warning(
                "Encoder queue full for %s, dropped %s frame(s). Consider using"
                " hardware encoding or increasing encoder_queue_maxsize.",
                video_key,
                count,
            )


def _fast_camera_encoder_thread_run(self) -> None:
    """Thread body patched onto LeRobot's camera encoder thread."""
    from lerobot.datasets.video_utils import _get_codec_options

    container = None
    output_stream = None
    stats_tracker = _FastVideoStats()
    frame_count = 0

    try:
        logging.getLogger("libav").setLevel(av.logging.WARNING)

        while True:
            try:
                frame_data = self.frame_queue.get(timeout=1)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue

            if frame_data is None:
                break

            if not isinstance(frame_data, np.ndarray):
                frame_data = np.asarray(frame_data)
            if frame_data.ndim == 3 and frame_data.shape[0] == 3:
                frame_data = frame_data.transpose(1, 2, 0)
            if frame_data.dtype != np.uint8:
                frame_data = np.clip(frame_data * 255.0, 0, 255).astype(np.uint8)
            elif not frame_data.flags.c_contiguous:
                frame_data = np.ascontiguousarray(frame_data)

            if container is None:
                height, width = frame_data.shape[:2]
                video_options = _get_codec_options(self.vcodec, self.g, self.crf, self.preset)
                if self.encoder_threads is not None:
                    if self.vcodec == "libsvtav1":
                        lp_param = f"lp={self.encoder_threads}"
                        if "svtav1-params" in video_options:
                            video_options["svtav1-params"] += f":{lp_param}"
                        else:
                            video_options["svtav1-params"] = lp_param
                    else:
                        video_options["threads"] = str(self.encoder_threads)
                Path(self.video_path).parent.mkdir(parents=True, exist_ok=True)
                container = av.open(str(self.video_path), "w")
                output_stream = container.add_stream(self.vcodec, self.fps, options=video_options)
                output_stream.pix_fmt = self.pix_fmt
                output_stream.width = width
                output_stream.height = height
                output_stream.time_base = Fraction(1, self.fps)

            video_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            video_frame.pts = frame_count
            video_frame.time_base = Fraction(1, self.fps)
            packet = output_stream.encode(video_frame)
            if packet:
                container.mux(packet)

            stats_tracker.update(frame_data)
            frame_count += 1

        if output_stream is not None:
            packet = output_stream.encode()
            if packet:
                container.mux(packet)

        if container is not None:
            container.close()

        av.logging.restore_default_callback()
        self.result_queue.put(("ok", stats_tracker.get_statistics()))

    except Exception as exc:  # pragma: no cover - exercised only in live encode threads
        if container is not None:
            with contextlib.suppress(Exception):
                container.close()
        self.result_queue.put(("error", str(exc)))


def apply_gpu_runtime_patches() -> None:
    """Patch LeRobot's hot runtime paths once per process."""
    global _PATCH_APPLIED, _ORIGINAL_COMPUTE_EPISODE_STATS
    if _PATCH_APPLIED:
        return

    import lerobot.datasets.compute_stats as compute_stats_mod
    import lerobot.datasets.dataset_writer as dataset_writer_mod
    import lerobot.datasets.video_utils as video_utils_mod

    if _ORIGINAL_COMPUTE_EPISODE_STATS is None:
        _ORIGINAL_COMPUTE_EPISODE_STATS = dataset_writer_mod.compute_episode_stats

    dataset_writer_mod.compute_episode_stats = compute_fast_episode_stats
    compute_stats_mod.compute_episode_stats = compute_fast_episode_stats
    video_utils_mod.StreamingVideoEncoder.feed_frame = _fast_feed_frame
    video_utils_mod._CameraEncoderThread.run = _fast_camera_encoder_thread_run
    _PATCH_APPLIED = True
