from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).with_name("archive_training_checkpoints.sh")


def make_checkpoint(runs: Path, job: str, step: str, value: str) -> Path:
    checkpoint = runs / job / "output" / "checkpoints" / step
    (checkpoint / "pretrained_model").mkdir(parents=True)
    (checkpoint / "training_state").mkdir()
    (checkpoint / "pretrained_model" / "model.safetensors").write_text(value)
    (checkpoint / "training_state" / "training_step.json").write_text(
        f'{{"step": {int(step)}}}\n'
    )
    return checkpoint


def test_archives_completed_checkpoints_and_retains_only_latest(tmp_path: Path):
    runs = tmp_path / "runs"
    archive = tmp_path / "archive"
    first = make_checkpoint(runs, "job-a", "002500", "first")
    second = make_checkpoint(runs, "job-a", "005000", "second")

    result = subprocess.run(
        [str(SCRIPT), str(runs), str(archive), "--once"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "ARCHIVED job=job-a checkpoint=002500" in result.stdout
    assert "ARCHIVED job=job-a checkpoint=005000" in result.stdout
    assert not first.exists()
    assert second.exists()
    assert (archive / "job-a" / "002500" / "ARCHIVE_COMPLETE").is_file()
    assert (archive / "job-a" / "005000" / "ARCHIVE_COMPLETE").is_file()
    assert (archive / "job-a" / "002500" / "pretrained_model" / "model.safetensors").read_text() == "first"


def test_ignores_checkpoint_until_training_state_is_complete(tmp_path: Path):
    runs = tmp_path / "runs"
    archive = tmp_path / "archive"
    incomplete = runs / "job-b" / "output" / "checkpoints" / "002500"
    (incomplete / "pretrained_model").mkdir(parents=True)
    (incomplete / "pretrained_model" / "model.safetensors").write_text("partial")

    subprocess.run(
        [str(SCRIPT), str(runs), str(archive), "--once"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert incomplete.exists()
    assert not (archive / "job-b" / "002500").exists()
