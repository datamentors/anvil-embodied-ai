"""Tests for GPU runtime fast-path helpers."""

import numpy as np

from mcap_converter.core.gpu_runtime import compute_fast_episode_stats


def test_compute_fast_episode_stats_preserves_vector_shapes_and_keys():
    episode_data = {
        "observation.state": np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float32),
        "action": np.array([0.25, 0.75, 1.25], dtype=np.float32),
        "task": ["teleop", "teleop", "teleop"],
    }
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,)},
        "action": {"dtype": "float32", "shape": (1,)},
        "task": {"dtype": "string", "shape": ()},
    }

    stats = compute_fast_episode_stats(episode_data, features)

    assert set(stats["observation.state"]) == {
        "min",
        "max",
        "mean",
        "std",
        "count",
        "q01",
        "q10",
        "q50",
        "q90",
        "q99",
    }
    assert stats["observation.state"]["mean"].shape == (2,)
    assert stats["observation.state"]["count"].tolist() == [3]

    assert stats["action"]["mean"].shape == (1,)
    assert stats["action"]["min"].shape == (1,)
    assert stats["action"]["count"].tolist() == [3]
