"""Tests for the GPU-optimized MCAP converter path."""

from pathlib import Path

from mcap_converter.cli.convert import build_parser
from mcap_converter.core.writer import LeRobotWriter


def test_gpu_parser_uses_gpu_friendly_defaults():
    parser = build_parser(profile="gpu")

    args = parser.parse_args(["-i", "/tmp/input", "-o", "/tmp/out"])

    assert args.vcodec == "auto"
    assert args.debug_plot_episodes == 0


def test_writer_create_forwards_streaming_encoder_args(monkeypatch, tmp_path):
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("mcap_converter.core.writer.LeRobotDataset.create", fake_create)

    writer = LeRobotWriter(
        output_dir=str(tmp_path / "dataset"),
        repo_id="test/gpu-create",
        vcodec="auto",
        streaming_encoding=True,
        encoder_queue_maxsize=120,
        encoder_threads=2,
    )

    writer.create_dataset({"": ["joint1"]}, ["waist"])

    assert captured["vcodec"] == "auto"
    assert captured["streaming_encoding"] is True
    assert captured["encoder_queue_maxsize"] == 120
    assert captured["encoder_threads"] == 2


def test_writer_resume_forwards_streaming_encoder_args(monkeypatch, tmp_path):
    captured = {}

    def fake_resume(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("mcap_converter.core.writer.LeRobotDataset.resume", fake_resume)
    monkeypatch.setattr("mcap_converter.core.writer._patch_resume_video_continuation", lambda dataset: None)

    writer = LeRobotWriter(
        output_dir=str(Path(tmp_path) / "dataset"),
        repo_id="test/gpu-resume",
        vcodec="auto",
        streaming_encoding=True,
        encoder_queue_maxsize=120,
        encoder_threads=2,
    )

    writer.load_dataset_for_writing()

    assert captured["root"] == str(Path(tmp_path) / "dataset")
    assert captured["vcodec"] == "auto"
    assert captured["streaming_encoding"] is True
    assert captured["encoder_queue_maxsize"] == 120
    assert captured["encoder_threads"] == 2
