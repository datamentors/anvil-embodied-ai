"""Tests for GPU parallel episode planning helpers."""

from pathlib import Path

from mcap_convert_gpu.cli.convert import plan_episode_shards, resolve_parallel_episode_workers


def test_plan_episode_shards_preserves_order_and_balances_sizes():
    files = [Path(f"/tmp/{idx:04d}.mcap") for idx in range(5)]

    shards = plan_episode_shards(files, worker_count=3)

    assert [[path.name for path in shard] for shard in shards] == [
        ["0000.mcap", "0001.mcap"],
        ["0002.mcap", "0003.mcap"],
        ["0004.mcap"],
    ]


def test_resolve_parallel_episode_workers_auto_caps_by_episode_count(monkeypatch):
    monkeypatch.setattr("mcap_convert_gpu.cli.convert.os.cpu_count", lambda: 28)

    assert resolve_parallel_episode_workers(0, episode_count=5) == 1
    assert resolve_parallel_episode_workers(0, episode_count=12) == 3
    assert resolve_parallel_episode_workers(0, episode_count=2) == 1
    assert resolve_parallel_episode_workers(7, episode_count=3) == 3
    assert resolve_parallel_episode_workers(1, episode_count=9) == 1
