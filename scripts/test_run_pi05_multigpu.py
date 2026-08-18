"""Contract checks for the reusable Pi0.5 multi-GPU launcher."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).with_name("run_pi05_multigpu.sh")


def test_launcher_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_launcher_contains_no_experiment_paths_or_task() -> None:
    payload = SCRIPT.read_text()

    assert "/home/datamentors/" not in payload
    assert "1000plus" not in payload
    assert "Pick up the envelope" not in payload
    assert "AFO N10" not in payload


def test_launcher_reads_training_contract_from_trainready_marker() -> None:
    payload = SCRIPT.read_text()

    assert 'marker = json.loads((root / "TRAIN_READY.json").read_text())' in payload
    assert 'facts["action_type"] != "absolute"' in payload
    assert 'facts["afo_lookahead_frames"]' in payload
    assert 'facts.get("task_prompts", [])' in payload


def test_preflight_uses_marker_prompt_and_configured_global_batch(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "meta" / "episodes").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}")
    (dataset / "meta" / "stats.json").write_text("{}")
    (dataset / "TRAIN_READY.json").write_text(
        json.dumps(
            {
                "facts": {
                    "action_type": "absolute",
                    "afo_lookahead_frames": 7,
                    "task_prompts": ["Move the object to the goal"],
                }
            }
        )
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(10)),
                "length": [64] * 10,
            }
        ),
        dataset / "meta" / "episodes" / "episodes.parquet",
    )

    source = tmp_path / "source"
    shim = source / "scripts" / "_ddp_shim" / "sitecustomize.py"
    shim.parent.mkdir(parents=True)
    shim.write_text("")
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python_wrapper = venv_bin / "python"
    python_wrapper.write_text(f'#!/usr/bin/env bash\nexec {Path(sys.executable)!s} "$@"\n')
    python_wrapper.chmod(0o755)
    for executable in ("accelerate", "anvil-trainer"):
        path = venv_bin / executable
        path.write_text("#!/usr/bin/env bash\nexit 0\n")
        path.chmod(0o755)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/usr/bin/env bash\nexit 0\n")
    nvidia_smi.chmod(0o755)
    hf_cache = tmp_path / "hf-cache"
    hf_cache.mkdir()

    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DATASET_ROOT": str(dataset),
        "ANVIL_TRAIN_SOURCE": str(source),
        "VENV": str(venv_bin.parent),
        "HF_CACHE": str(hf_cache),
        "RUN_ROOT": str(tmp_path / "runs"),
        "CUDA_DEVICES": "2,5",
        "BATCH_PER_GPU": "8",
        "PREFLIGHT_ONLY": "1",
    }
    result = subprocess.run(
        [str(SCRIPT), "full_vlm"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Batch: 8/GPU x 2 = 16" in result.stdout
    assert "Task: Move the object to the goal" in result.stdout
    assert "AFO lookahead: 7 frames" in result.stdout
    assert "Training preflight passed; no process was started." in result.stdout
