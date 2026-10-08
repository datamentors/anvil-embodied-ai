"""
MCAP to LeRobot Dataset Converter (Modular Version)

Uses extracted core modules for cleaner, testable code.
"""

import argparse
import concurrent.futures
import contextlib
import json
import multiprocessing
import os
import shutil
import stat
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List

import av
import huggingface_hub
import yaml
from anvil_shared.provenance import git_provenance
from rich.console import Console, Group
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from mcap_converter import (
    ConfigLoader,
    DataConfig,
    LeRobotWriter,
    McapReader,
)
from mcap_converter.cli.mcap_valid import default_report_paths
from mcap_converter.core.episode_labels import (
    build_label_table,
    install_episode_metadata_injector,
    resolve_label_keys,
)
from mcap_converter.core.extractor import BufferedStreamExtractor
from mcap_converter.core.quality import SEVERITY_CRITICAL, SEVERITY_PASS, SEVERITY_WARNING
from mcap_converter.core.reader import snap_fps

console = Console()


_PROFILE_DEFAULTS = {
    "standard": {
        "title": "[bold]MCAP to LeRobot Dataset Converter",
        "description": "Convert MCAP recordings to LeRobot v3.0 dataset format",
        "examples": """\
examples:
  mcap-convert -i data/raw/my-session -o data/datasets --config configs/mcap_converter/openarm_bimanual.yaml
  # output goes to data/datasets/my-session/

  mcap-convert -i data/raw/my-session -o data/datasets --vcodec libsvtav1
  mcap-convert -i data/raw/my-session -o data/datasets --fps 15 --push-to-hub
  mcap-convert -i data/raw/my-session -o data/datasets --max-episodes 5
  mcap-convert -i data/raw/my-session -o data/datasets --resume
  mcap-convert -i data/raw/my-session -o data/datasets  # default: critical episodes skipped automatically
  mcap-convert -i data/raw/my-session -o data/datasets --include-flagged critical  # convert everything, even critical episodes
""",
        "default_vcodec": "h264",
        "default_debug_plot_episodes": 5,
        "streaming_encoding": False,
        "encoder_queue_maxsize": 30,
        "encoder_threads": None,
        "progress_update_every": 1,
        "parallel_episode_workers": 1,
        "label": "standard",
    },
    "gpu": {
        "title": "[bold]MCAP to LeRobot Dataset Converter (GPU Path)",
        "description": "Convert MCAP recordings to LeRobot datasets using the GPU-optimized path",
        "examples": """\
examples:
  mcap-convert-gpu -i data/raw/my-session -o data/datasets --config configs/mcap_converter/openarm_bimanual.yaml
  # output goes to data/datasets/my-session/

  mcap-convert-gpu -i data/raw/my-session -o data/datasets --vcodec auto
  mcap-convert-gpu -i data/raw/my-session -o data/datasets --fps 15 --push-to-hub
  mcap-convert-gpu -i data/raw/my-session -o data/datasets --max-episodes 5
  mcap-convert-gpu -i data/raw/my-session -o data/datasets --resume
  mcap-convert-gpu -i data/raw/my-session -o data/datasets --include-flagged critical
""",
        "default_vcodec": "auto",
        "default_debug_plot_episodes": 0,
        "streaming_encoding": True,
        "encoder_queue_maxsize": 240,
        "encoder_threads": None,
        "progress_update_every": 64,
        "parallel_episode_workers": 0,
        "label": "gpu",
    },
}


def log(message: str) -> None:
    """Print a timestamped log message, left-aligned."""
    ts = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim][{ts}][/dim] {message}")


@contextlib.contextmanager
def suppress_fd_output():
    """Suppress stdout/stderr at the file descriptor level (catches C/ffmpeg output)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def collect_mcap_files(input_dir: str) -> List[Path]:
    """Recursively collect all MCAP files under input directory"""
    mcap_paths = []
    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            if file.endswith(".mcap"):
                mcap_paths.append(Path(root) / file)
    return sorted(mcap_paths)


def detect_gpu_count() -> int:
    """Return the number of CUDA-visible GPUs, or 0 if none/torch unavailable.

    Respects CUDA_VISIBLE_DEVICES if the caller's own environment already
    restricts it. Used to round-robin shard workers across physical GPUs —
    without this, every worker's NVENC session would default to GPU 0.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


def resolve_parallel_episode_workers(requested: int, episode_count: int) -> int:
    """Resolve how many episode workers to use.

    `requested=0` means auto. Auto stays conservative because one workstation GPU
    can handle several concurrent episodes, but too many workers just add merge and
    I/O overhead.
    """
    if episode_count <= 1:
        return 1
    if requested < 0:
        raise ValueError("parallel_episode_workers must be >= 0")
    if requested == 1:
        return 1
    if requested == 0:
        if episode_count < 8:
            return 1
        cpu_count = os.cpu_count() or 1
        auto_workers = max(2, cpu_count // 8) if cpu_count >= 8 else 1
        gpu_count = detect_gpu_count()
        cap = max(4, gpu_count) if gpu_count > 0 else 4
        return max(1, min(cap, episode_count, auto_workers))
    return max(1, min(episode_count, requested))


def plan_episode_shards(mcap_files: List[Path], worker_count: int) -> List[List[Path]]:
    """Split episodes into contiguous shards while preserving original order."""
    if worker_count <= 1 or len(mcap_files) <= 1:
        return [list(mcap_files)]

    total = len(mcap_files)
    base = total // worker_count
    remainder = total % worker_count
    shards: list[list[Path]] = []
    start = 0
    for idx in range(worker_count):
        shard_len = base + (1 if idx < remainder else 0)
        if shard_len <= 0:
            continue
        end = start + shard_len
        shards.append(list(mcap_files[start:end]))
        start = end
    return shards


def _codec_available(codec_name: str) -> bool:
    """Return whether PyAV/FFmpeg can open this encoder on the current machine."""
    try:
        av.codec.Codec(codec_name, "w")
        return True
    except Exception:
        return False


def resolve_profile_vcodec(profile: str, requested_vcodec: str) -> str:
    """Resolve a profile-specific preferred codec without breaking portability.

    For the GPU path, prefer HEVC NVENC when available because it benchmarked
    faster than the current auto-selected H.264 NVENC on the workstation.
    """
    if profile != "gpu":
        return requested_vcodec
    if requested_vcodec != "auto":
        return requested_vcodec
    if _codec_available("hevc_nvenc"):
        return "hevc_nvenc"
    return requested_vcodec


# Single source of truth for severity ordering, shared with core/quality.py's
# SEVERITY_PASS/WARNING/CRITICAL constants (rather than re-hardcoding the same
# three strings here) so a future rename can't drift between the two files.
_SEVERITY_ORDER = [SEVERITY_PASS, SEVERITY_WARNING, SEVERITY_CRITICAL]


def resolve_quality_skip_paths(quality_report_path: str | None, include_flagged: str) -> dict:
    """
    Read a mcap-valid JSON report and return {resolved_path: severity} for the
    episodes that fall ABOVE the --include-flagged threshold and should be
    skipped during conversion.

    include_flagged is an inclusive threshold, not an exclusion list: "pass"
    converts only pass-severity episodes (skips warning+critical); "warning"
    (the CLI default) also converts warning episodes, skipping only critical;
    "critical" converts everything, skipping nothing.
    """
    if quality_report_path is None:
        return {}

    with open(quality_report_path) as f:
        payload = json.load(f)

    threshold_idx = _SEVERITY_ORDER.index(include_flagged)
    skip_severities = set(_SEVERITY_ORDER[threshold_idx + 1 :])
    return {
        ep["path"]: ep["severity"]
        for ep in payload.get("episodes", [])
        if ep["severity"] in skip_severities
    }


def parse_episode_index_spec(spec: str, total_episodes: int) -> set:
    """
    Parse a 1-based episode index spec into a concrete set of indices.

    Colon ranges follow Python slice convention: the end is EXCLUSIVE, e.g.
    "1:4" selects episodes 1, 2, 3 (not 4) — same as Python's range(1, 4).
    An omitted start defaults to 1; an omitted end reaches the actual last
    episode inclusively (there's nothing to exclude when no end is given).
    """
    result: set = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            start_str, end_str = token.split(":", 1)
            start_str, end_str = start_str.strip(), end_str.strip()
            try:
                start = int(start_str) if start_str else 1
                end = int(end_str) if end_str else total_episodes + 1
            except ValueError:
                raise ValueError(f"invalid episode range token: '{token}'")
            if start >= end:
                raise ValueError(f"invalid range '{token}': start must be less than end (end is exclusive)")
            if start < 1 or end > total_episodes + 1:
                raise ValueError(f"range '{token}' out of bounds — episodes are numbered 1 to {total_episodes}")
            result.update(range(start, end))
        else:
            try:
                idx = int(token)
            except ValueError:
                raise ValueError(f"invalid episode index token: '{token}'")
            if not (1 <= idx <= total_episodes):
                raise ValueError(f"episode index {idx} out of range (1-{total_episodes})")
            result.add(idx)
    return result


def quick_scan_joint_names(mcap_path: str, config: DataConfig) -> dict:
    """
    Quick scan to extract joint names from first JointState message.

    Only reads the first message, so memory-efficient for large files.

    In leader-follower mode: parses joint names to find observation (follower) joints.
    In quest teleop mode: all joints in the JointState topic are observations,
    so we group by arm without filtering by source/role prefix.

    Returns:
        Dictionary mapping robot prefix to joint names:
        - {"right": ["joint1", ...], "left": [...]} for multi-robot
        - {"": ["joint1", ...]} for single robot
        Joint names are extracted from the observation role.
    """
    reader = McapReader(mcap_path)
    joint_pattern = config.joint_name_pattern
    sep = joint_pattern.separator
    quest_mode = bool(config.action_topics)

    for message in reader.read_messages(topics=[config.robot_state_topic]):
        ros_msg = message.ros_msg

        # Group joint names by robot prefix
        robot_joints: dict = {}  # {robot_prefix: [joint_ids]}

        for joint_name in ros_msg.name:
            if quest_mode:
                # Quest teleop mode: all joints are observations (no leader prefix).
                # Parse arm identifier and joint_id directly.
                # Joint names are like "follower_l_joint1" — still use source
                # prefix to strip it, then extract arm and joint_id.
                remaining = joint_name
                robot = ""

                # Try to strip known source prefixes
                for prefix in joint_pattern.role_prefix.keys():
                    if joint_name.startswith(prefix + sep):
                        remaining = joint_name[len(prefix) + len(sep) :]
                        break

                # Extract robot prefix and joint_id
                parts = remaining.split(sep, 1)
                if parts and parts[0] in joint_pattern.robot_prefix:
                    robot = joint_pattern.robot_prefix[parts[0]]
                    joint_id = parts[1] if len(parts) > 1 else parts[0]
                else:
                    robot = ""
                    joint_id = remaining

                if robot not in robot_joints:
                    robot_joints[robot] = []
                robot_joints[robot].append(joint_id)
            else:
                # Leader-follower mode: only extract observation (follower) joints
                role = None
                robot = ""
                remaining = ""

                for prefix, role_name in joint_pattern.role_prefix.items():
                    if joint_name.startswith(prefix + sep):
                        role = role_name
                        remaining = joint_name[len(prefix) + len(sep) :]
                        break

                if role != "observation":
                    continue

                # Extract robot prefix and joint_id
                parts = remaining.split(sep, 1)
                if parts and parts[0] in joint_pattern.robot_prefix:
                    robot = joint_pattern.robot_prefix[parts[0]]
                    joint_id = parts[1] if len(parts) > 1 else parts[0]
                else:
                    robot = ""
                    joint_id = remaining

                if robot not in robot_joints:
                    robot_joints[robot] = []
                robot_joints[robot].append(joint_id)

        if robot_joints:
            # Sort each arm's joint list for canonical ordering
            for robot in robot_joints:
                robot_joints[robot] = sorted(robot_joints[robot])
            return robot_joints

    return {}


def _ensure_output_readable(output_dir: str) -> None:
    """
    Ensure every file/directory in the converted dataset is readable by
    other users, not just the owner.

    Works around a known issue where lerobot's video encoder (PyAV/
    libavformat, via encode_video_frames() in lerobot/datasets/
    video_utils.py) writes video files with 0600 permissions, bypassing the
    process umask. This breaks any tool that needs to read the dataset as a
    different user/UID than the one that ran the conversion — e.g. a
    different local user or service account reading the dataset later.
    """
    root = Path(output_dir)
    for path in root.rglob("*"):
        try:
            current_mode = stat.S_IMODE(path.stat().st_mode)
            if path.is_dir():
                # ensure r+w+x for owner, r+x for group/other (on top of whatever's already set)
                os.chmod(path, current_mode | 0o755)
            else:
                # ensure r+w for owner, r for group/other
                os.chmod(path, current_mode | 0o644)
        except OSError:
            # best-effort: don't let a permission fix-up failure crash the
            # whole conversion; the dataset is still usable by the owner.
            continue


def _write_effective_conversion_config(config: DataConfig, destination: str | Path) -> None:
    """Persist the configuration actually used, including CLI overrides."""
    config_to_save = asdict(config)
    config_to_save["joint_names"] = config_to_save.pop("joint_name_pattern")
    if not config_to_save["robot_state_topics"]:
        config_to_save.pop("robot_state_topics")
    if not config_to_save["motor_feature_mapping"]:
        config_to_save.pop("motor_feature_mapping")
    with open(destination, "w") as config_file:
        yaml.safe_dump(config_to_save, config_file, sort_keys=False)


def _copy_conversion_config_from_shard(shard_output_dir: Path, output_dir: Path) -> None:
    """Keep the final merged dataset's conversion config alongside the data."""
    src = shard_output_dir / "conversion_config.yaml"
    dst = output_dir / "conversion_config.yaml"
    if src.is_file():
        shutil.copy(src, dst)


def _merge_parallel_shards(
    shard_results: list[dict],
    *,
    output_dir: str,
    repo_id: str,
):
    """Merge shard datasets back into one final dataset in shard order."""
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ordered = sorted(shard_results, key=lambda item: item["shard_index"])
    shard_datasets = [
        LeRobotDataset(result["repo_id"], root=Path(result["output_dir"])) for result in ordered
    ]
    merged = merge_datasets(
        shard_datasets,
        output_repo_id=repo_id,
        output_dir=Path(output_dir),
    )
    _copy_conversion_config_from_shard(Path(ordered[0]["output_dir"]), Path(output_dir))
    return merged


def _convert_shard_worker(
    *,
    shard_index: int,
    shard_paths: list[str],
    input_dir: str,
    output_dir: str,
    repo_id: str,
    robot_type: str,
    fps: int,
    tolerance_s: float,
    task: str,
    config_dict: dict,
    buffer_seconds: float,
    config_path: str | None,
    vcodec: str,
    streaming_encoding: bool,
    encoder_queue_maxsize: int,
    encoder_threads: int | None,
    progress_update_every: int,
    gpu_id: int | None = None,
    extra_episode_keys: tuple = (),
) -> dict:
    """Worker process entrypoint for one shard of episodes.

    gpu_id, when set, pins this shard to a single physical GPU by restricting
    CUDA_VISIBLE_DEVICES before any CUDA-touching import (torch, PyAV/NVENC)
    happens in convert_session() below. Each worker runs in its own spawned
    process (fresh interpreter), so this env var only ever affects this shard.
    Without it, every worker's NVENC session lands on physical GPU 0 by
    default — spreading load across a 4-GPU box requires this.
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = Path(output_dir).parent / f"shard-{shard_index:03d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    shard_start = time.time()
    with open(log_path, "w") as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        config = ConfigLoader.from_dict(config_dict)
        convert_session(
            input_dir=input_dir,
            output_dir=output_dir,
            repo_id=repo_id,
            robot_type=robot_type,
            fps=fps,
            tolerance_s=tolerance_s,
            task=task,
            config=config,
            buffer_seconds=buffer_seconds,
            config_path=config_path,
            vcodec=vcodec,
            resume_from=0,
            max_episodes=None,
            mcap_files=[Path(path) for path in shard_paths],
            debug_plot_episodes=0,
            quality_skip_paths=None,
            skip_episode_indices=None,
            streaming_encoding=streaming_encoding,
            encoder_queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
            progress_update_every=progress_update_every,
            use_live_progress=False,
            parallel_episode_workers=1,
            extra_episode_keys=extra_episode_keys,
        )

    return {
        "shard_index": shard_index,
        "output_dir": output_dir,
        "repo_id": repo_id,
        "log_path": str(log_path),
        "episodes": len(shard_paths),
        "elapsed_s": time.time() - shard_start,
        "gpu_id": gpu_id,
    }


def convert_session(
    input_dir: str,
    output_dir: str,
    repo_id: str,
    robot_type: str = "anvil_openarm",
    fps: int = 30,
    tolerance_s: float = 1e-3,
    task: str = "manipulation",
    config: DataConfig = None,
    buffer_seconds: float = 5.0,
    config_path: str = None,
    vcodec: str = "h264",
    resume_from: int = 0,
    max_episodes: int = None,
    mcap_files: List[Path] = None,
    debug_plot_episodes: int = 5,
    quality_skip_paths: dict | None = None,
    skip_episode_indices: set | None = None,
    streaming_encoding: bool = False,
    encoder_queue_maxsize: int = 30,
    encoder_threads: int | None = None,
    progress_update_every: int = 1,
    use_live_progress: bool = True,
    parallel_episode_workers: int = 1,
    extra_episode_keys: tuple = (),
):
    """
    Convert MCAP session to LeRobot dataset

    Args:
        input_dir: Directory containing MCAP files
        output_dir: Output directory for dataset
        repo_id: HuggingFace repository ID
        robot_type: Robot type identifier
        fps: Video frames per second
        tolerance_s: Time synchronization tolerance
        task: Task name for the dataset
        config: Data configuration
        buffer_seconds: Buffer window for time alignment in seconds (default: 5.0)
        config_path: Path to the conversion config YAML file (for copying to output)
        vcodec: Video codec for encoding ("h264", "hevc", or "libsvtav1")
        streaming_encoding: If True, use LeRobot's streaming video encoder
        encoder_queue_maxsize: Queue depth per camera for streaming video encoding
        encoder_threads: Threads per encoder instance (or None for codec default)
        progress_update_every: Throttle progress callback updates to every N frames
        use_live_progress: Enable Rich live progress rendering
        parallel_episode_workers: Number of episode shards to run in parallel (0 = auto)
    """
    session_start_time = time.time()

    if config is None:
        config = ConfigLoader.get_default()

    if streaming_encoding:
        from mcap_converter.core.gpu_runtime import apply_gpu_runtime_patches

        apply_gpu_runtime_patches()

    # Find all MCAP files (use pre-collected list if provided)
    if mcap_files is None:
        mcap_files = collect_mcap_files(input_dir)
    if not mcap_files:
        raise FileNotFoundError(f"No .mcap files found in {input_dir}")

    if max_episodes is not None:
        mcap_files = mcap_files[:max_episodes]
        log(f"Found [bold]{len(mcap_files)}[/bold] MCAP files (limited to first {max_episodes})")
    else:
        log(f"Found [bold]{len(mcap_files)}[/bold] MCAP files")
    log(f"Buffered streaming (buffer={buffer_seconds}s)")

    effective_parallel_workers = resolve_parallel_episode_workers(
        parallel_episode_workers,
        len(mcap_files),
    )
    can_parallelize = (
        streaming_encoding
        and effective_parallel_workers > 1
        and resume_from == 0
    )

    if can_parallelize:
        included_mcap_files: list[Path] = []
        skip_paths = quality_skip_paths or {}
        for episode_idx, mcap_path in enumerate(mcap_files):
            if quality_severity := skip_paths.get(str(mcap_path.resolve())):
                color = "red" if quality_severity == "critical" else "yellow"
                console.print(
                    f"  [{color}]↷ [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}"
                    f"  skipped (quality: {quality_severity})[/{color}]"
                )
                continue
            if skip_episode_indices and (episode_idx + 1) in skip_episode_indices:
                console.print(
                    f"  [cyan]↷ [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}"
                    f"  skipped (manual index)[/cyan]"
                )
                continue
            included_mcap_files.append(mcap_path)

        if len(included_mcap_files) > 1:
            shards = plan_episode_shards(included_mcap_files, effective_parallel_workers)
            shard_root_parent = Path(output_dir).parent
            shard_root_parent.mkdir(parents=True, exist_ok=True)
            shard_root = Path(
                tempfile.mkdtemp(
                    prefix=f".{Path(output_dir).name}.parallel-",
                    dir=str(shard_root_parent),
                )
            )
            log(
                f"GPU parallel mode: [bold]{len(shards)}[/bold] shards across "
                f"[bold]{effective_parallel_workers}[/bold] workers"
            )

            gpu_count = detect_gpu_count()
            if gpu_count > 0:
                log(f"Pinning shards round-robin across [bold]{gpu_count}[/bold] GPU(s)")

            config_dict = asdict(config)
            shard_results: list[dict] = []
            try:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=len(shards),
                    mp_context=multiprocessing.get_context("spawn"),
                ) as executor:
                    future_map = {}
                    for shard_index, shard_files in enumerate(shards):
                        shard_output_dir = shard_root / f"shard-{shard_index:03d}"
                        shard_repo_id = f"{repo_id}-shard-{shard_index:03d}"
                        shard_gpu_id = (shard_index % gpu_count) if gpu_count > 0 else None
                        future = executor.submit(
                            _convert_shard_worker,
                            shard_index=shard_index,
                            shard_paths=[str(path) for path in shard_files],
                            input_dir=input_dir,
                            output_dir=str(shard_output_dir),
                            repo_id=shard_repo_id,
                            robot_type=robot_type,
                            fps=fps,
                            tolerance_s=tolerance_s,
                            task=task,
                            config_dict=config_dict,
                            buffer_seconds=buffer_seconds,
                            config_path=config_path,
                            vcodec=vcodec,
                            streaming_encoding=streaming_encoding,
                            encoder_queue_maxsize=encoder_queue_maxsize,
                            encoder_threads=encoder_threads,
                            progress_update_every=progress_update_every,
                            gpu_id=shard_gpu_id,
                            extra_episode_keys=extra_episode_keys,
                        )
                        future_map[future] = (shard_index, len(shard_files))

                    for future in concurrent.futures.as_completed(future_map):
                        shard_index, shard_episodes = future_map[future]
                        result = future.result()
                        shard_results.append(result)
                        gpu_label = (
                            f"  [dim](gpu {result['gpu_id']})[/dim]" if result.get("gpu_id") is not None else ""
                        )
                        console.print(
                            f"  [green]✓[/green] shard {shard_index + 1}/{len(shards)}"
                            f"  {shard_episodes} episode(s)"
                            f"  {format_duration(result['elapsed_s'])}"
                            f"{gpu_label}"
                        )

                dataset = _merge_parallel_shards(
                    shard_results,
                    output_dir=output_dir,
                    repo_id=repo_id,
                )
                _ensure_output_readable(output_dir)

                if dataset.meta.total_frames > 0 and debug_plot_episodes > 0:
                    from mcap_converter.utils.debug_plot import plot_conversion_debug

                    with console.status("[bold]Generating debug plots..."):
                        plot_conversion_debug(
                            output_dir,
                            n_episodes=debug_plot_episodes,
                            action_from_observation_n=config.action_from_observation_n,
                        )
                    log(f"Debug plots saved to [dim]{output_dir}/debug_plots/[/dim]")

                total_time = time.time() - session_start_time
                avg_episode_time = (
                    total_time / dataset.meta.total_episodes if dataset.meta.total_episodes else 0
                )
                fps_actual = dataset.meta.total_frames / total_time if total_time > 0 else 0

                summary = Table(show_header=False, box=None, padding=(0, 2))
                summary.add_column(style="bold")
                summary.add_column()
                summary.add_row("Episodes", str(dataset.meta.total_episodes))
                summary.add_row("Total frames", str(dataset.meta.total_frames))
                summary.add_row("Location", output_dir)
                summary.add_row("Shards", str(len(shards)))

                timing = Table(show_header=False, box=None, padding=(0, 2))
                timing.add_column(style="bold")
                timing.add_column()
                timing.add_row("Total time", format_duration(total_time))
                timing.add_row("Avg per episode", format_duration(avg_episode_time))
                timing.add_row("Processing rate", f"{fps_actual:.1f} frames/sec")

                report = Panel(
                    Group(summary, "", timing),
                    title="[bold green]LeRobot Dataset Created Successfully",
                    border_style="green",
                    padding=(1, 2),
                )
                console.print(report)
                return dataset
            finally:
                shutil.rmtree(shard_root, ignore_errors=True)

    # Initialize writer (quiet — Rich handles output)
    writer = LeRobotWriter(
        output_dir=output_dir,
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        config=config,
        vcodec=vcodec,
        streaming_encoding=streaming_encoding,
        encoder_queue_maxsize=encoder_queue_maxsize,
        encoder_threads=encoder_threads,
        quiet=True,
    )

    # Get joint names
    log(f"Quick scan for joint names: [dim]{mcap_files[0]}[/dim]")
    joint_names = quick_scan_joint_names(str(mcap_files[0]), config)
    if not joint_names:
        raise ValueError("Cannot get joint names from reference MCAP (no observation joints found)")

    # Log detected robot mode
    robots = [r for r in joint_names.keys() if r]
    total_joints = sum(len(v) for v in joint_names.values())
    quest_mode = bool(config.action_topics)
    teleop_label = "[bold magenta]quest teleop[/bold magenta]" if quest_mode else "[bold cyan]leader-follower[/bold cyan]"
    if robots:
        log(f"Detected [bold cyan]bimanual[/bold cyan] robot ({teleop_label}): {robots}")
        for robot in sorted(robots):
            log(f"  {robot}: {joint_names[robot]}")
    else:
        log(f"Detected [bold cyan]single-arm[/bold cyan] robot ({teleop_label})")
        log(f"  joints: {joint_names.get('', [])}")
    log(f"Total joints: [bold]{total_joints}[/bold] (observation + action)")
    if quest_mode:
        for topic, topic_cfg in config.action_topics.items():
            log(f"  Action topic ({topic_cfg.arm}): [dim]{topic}[/dim]")

    # Get camera names
    camera_names = list(config.camera_topic_mapping.values())
    if not camera_names:
        raise ValueError("No camera images available, cannot create dataset image features")
    log(f"Cameras: {camera_names}")

    # Create or load dataset
    if resume_from > 0:
        dataset = writer.load_dataset_for_writing()
        log(f"Loaded existing dataset ({resume_from} episodes already converted)")
    else:
        dataset = writer.create_dataset(
            joint_names=joint_names,
            camera_names=camera_names,
        )

    # Carry recorder labels (envelope size/facing side, ...) into meta/episodes as
    # columns, plus provenance for which MCAP produced which episode. The key set
    # is fixed once per run: LeRobot flushes episode metadata through pyarrow and
    # a key present on only some episodes fails at flush time.
    label_keys = resolve_label_keys(mcap_files, output_dir, resume_from, tuple(extra_episode_keys or ()))
    label_table, incomplete_labels = build_label_table(mcap_files, label_keys)
    episode_extra = install_episode_metadata_injector(dataset)
    _labelled = [k for k in label_keys if not k.startswith("source_")]
    if _labelled:
        log(f"Episode labels: [bold]{', '.join(_labelled)}[/bold]")
        if incomplete_labels:
            log(
                f"  [yellow]{len(incomplete_labels)} episode(s) missing a label[/yellow] "
                f"([dim]{', '.join(incomplete_labels[:8])}"
                f"{' ...' if len(incomplete_labels) > 8 else ''}[/dim]) — written as empty"
            )
    else:
        log("Episode labels: [dim]none found in recorder metadata.json[/dim]")

    # Copy conversion config for inference generation during training (skip if resuming)
    conversion_config_dest = os.path.join(output_dir, "conversion_config.yaml")
    if resume_from > 0:
        log(f"Skipping config copy — using existing [dim]{conversion_config_dest}[/dim]")
    else:
        _write_effective_conversion_config(config, conversion_config_dest)
        log(f"Saved effective conversion config: [dim]{conversion_config_dest}[/dim]")

    # Append git provenance to conversion_config.yaml (skip when resuming — already present)
    if resume_from == 0:
        provenance = git_provenance()
        if provenance:
            with open(conversion_config_dest, "a") as _f:
                _f.write("\n# --- provenance ---\n")
                yaml.dump(provenance, _f, default_flow_style=False)

    # Process each MCAP file as one episode
    total_frames = 0
    episode_times = []
    episode_frame_counts = []
    episode_original_indices = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("{task.fields[status]}"),
        TextColumn("[dim]|[/dim]"),
        TimeElapsedColumn(),
        console=console,
        disable=not use_live_progress,
    ) as progress:
        overall_task = progress.add_task(
            "[bold blue]Converting episodes",
            total=len(mcap_files),
            status=f"{resume_from}/{len(mcap_files)} episodes",
        )

        skip_paths = quality_skip_paths or {}
        for episode_idx, mcap_path in enumerate(mcap_files):
            if episode_idx < resume_from:
                progress.advance(overall_task)
                progress.update(overall_task, status=f"{episode_idx + 1}/{len(mcap_files)} episodes [dim](skipped)[/dim]")
                console.print(f"  [dim]↷ [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}  skipped (already converted)[/dim]")
                continue

            quality_severity = skip_paths.get(str(mcap_path.resolve()))
            if quality_severity is not None:
                color = "red" if quality_severity == "critical" else "yellow"
                progress.advance(overall_task)
                progress.update(overall_task, status=f"{episode_idx + 1}/{len(mcap_files)} episodes [dim](skipped)[/dim]")
                console.print(
                    f"  [{color}]↷ [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}"
                    f"  skipped (quality: {quality_severity})[/{color}]"
                )
                continue

            if skip_episode_indices and (episode_idx + 1) in skip_episode_indices:
                progress.advance(overall_task)
                progress.update(overall_task, status=f"{episode_idx + 1}/{len(mcap_files)} episodes [dim](skipped)[/dim]")
                console.print(
                    f"  [cyan]↷ [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}"
                    f"  skipped (manual index)[/cyan]"
                )
                continue

            episode_start_time = time.time()

            episode_task = progress.add_task(
                f"  [dim]{mcap_path.name}[/dim]",
                total=None,
                status="starting...",
            )

            # Use buffered streaming for memory-efficient extraction (quiet — Rich handles output)
            frame_count = 0

            def on_frame_progress(count, _task=episode_task):
                nonlocal frame_count
                frame_count = count
                elapsed = time.time() - episode_start_time
                speed = count / elapsed if elapsed > 0 else 0
                progress.update(
                    _task,
                    completed=count,
                    status=f"[green]{count}[/green] frames [dim]({speed:.0f} f/s)[/dim]",
                )

            stream_extractor = BufferedStreamExtractor(
                config=config,
                buffer_seconds=buffer_seconds,
                fps=fps,
                quiet=True,
                progress_callback=on_frame_progress if use_live_progress else None,
                progress_every=max(1, progress_update_every),
            )

            corrupt_frame_error: Exception | None = None
            try:
                for frame in stream_extractor.extract_frames(str(mcap_path), task=task):
                    dataset.add_frame(frame)
                    if not use_live_progress:
                        frame_count += 1
            except ValueError as exc:
                corrupt_frame_error = exc

            if corrupt_frame_error is not None:
                # Discard any partially-buffered frames for this episode
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer(delete_images=True)
                progress.update(
                    episode_task,
                    total=1,
                    completed=1,
                    status=f"[red]skipped (corrupt frame: {corrupt_frame_error})[/red]",
                )
                progress.advance(overall_task)
                progress.update(
                    overall_task,
                    status=f"{episode_idx + 1}/{len(mcap_files)} episodes",
                )
                progress.remove_task(episode_task)
                console.print(
                    f"  [yellow]⚠[/yellow] [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}"
                    f"  [yellow]skipped (corrupt frame)[/yellow]"
                )
                episode_frame_counts.append(0)
                episode_times.append(time.time() - episode_start_time)
                episode_original_indices.append(episode_idx)
                log(
                    f"[yellow]⚠ Skipped episode {mcap_path.name} — corrupt frame: "
                    f"{corrupt_frame_error}[/yellow]"
                )
                continue

            if frame_count == 0:
                # Skip empty episodes — don't call save_episode on an empty buffer
                progress.update(
                    episode_task,
                    total=1,
                    completed=1,
                    status="[yellow]skipped (0 frames)[/yellow]",
                )
                progress.advance(overall_task)
                progress.update(
                    overall_task,
                    status=f"{episode_idx + 1}/{len(mcap_files)} episodes",
                )
                progress.remove_task(episode_task)
                console.print(
                    f"  [yellow]⚠[/yellow] [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}"
                    f"  [yellow]skipped (0 frames)[/yellow]"
                )
                episode_frame_counts.append(0)
                episode_times.append(time.time() - episode_start_time)
                episode_original_indices.append(episode_idx)
                continue

            for robot, counts in stream_extractor.get_action_fill_stats().items():
                filled = counts["hold_last"] + counts["fallback_to_observation"]
                if filled == 0 and counts["dropped"] == 0:
                    continue
                robot_label = robot or "action"
                dropped_suffix = (
                    f", [red]{counts['dropped']} dropped[/red]" if counts["dropped"] else ""
                )
                console.print(
                    f"    [yellow]↺[/yellow] {robot_label}: {counts['exact']} exact, "
                    f"{counts['hold_last']} hold-last, "
                    f"{counts['fallback_to_observation']} fallback-to-obs{dropped_suffix}"
                )

            # Save episode — suppress ffmpeg/libx264 noise
            progress.update(
                episode_task,
                status=f"[yellow]saving {frame_count} frames...[/yellow]",
            )
            episode_extra.clear()
            episode_extra.update(label_table[str(mcap_path)])
            with suppress_fd_output():
                dataset.save_episode()

            episode_time = time.time() - episode_start_time
            episode_times.append(episode_time)
            episode_frame_counts.append(frame_count)
            episode_original_indices.append(episode_idx)
            total_frames += frame_count

            # Mark episode done with green bar
            progress.update(
                episode_task,
                total=frame_count,
                completed=frame_count,
                status=f"[green]{frame_count} frames[/green] in {format_duration(episode_time)}",
            )
            progress.advance(overall_task)
            progress.update(
                overall_task,
                status=f"{episode_idx + 1}/{len(mcap_files)} episodes",
            )
            progress.remove_task(episode_task)
            ep_fps = frame_count / episode_time if episode_time > 0 else 0
            console.print(
                f"  [green]✓[/green] [{episode_idx + 1}/{len(mcap_files)}] {mcap_path.name}"
                f"  [green]{frame_count} frames[/green]"
                f"  {format_duration(episode_time)}"
                f"  {ep_fps:.0f} f/s"
            )

    # Check for all-empty conversion
    if total_frames == 0:
        console.print(
            "\n[bold red]ERROR: All episodes produced 0 frames.[/bold red]\n"
            "The extractor printed diagnostics above (scroll up).\n"
            "Common causes:\n"
            "  1. Camera topics in config don't match MCAP topics\n"
            "  2. Action topics don't exist in MCAP (quest mode)\n"
            "  3. Joint name prefixes don't match config source mapping\n"
            "  Run [bold]mcap-valid[/bold] on your MCAP to see all recorded topics and message types.\n"
        )
        return dataset

    # Finalize dataset
    with console.status("[bold]Finalizing dataset (metadata & cleanup)..."):
        with suppress_fd_output():
            writer.finalize(dataset)
        # lerobot's video encoder writes .mp4 files as 0600 (bypassing umask),
        # which blocks any reader running as a different UID. Fix up
        # permissions across the whole output tree.
        _ensure_output_readable(output_dir)

    # Debug plots: generated only when explicitly enabled for at least 1 episode
    if total_frames > 0 and debug_plot_episodes > 0:
        from mcap_converter.utils.debug_plot import plot_conversion_debug
        with console.status("[bold]Generating debug plots..."):
            plot_conversion_debug(
                output_dir,
                n_episodes=debug_plot_episodes,
                action_from_observation_n=config.action_from_observation_n,
            )
        log(f"Debug plots saved to [dim]{output_dir}/debug_plots/[/dim]")

    # Calculate timing statistics
    total_time = time.time() - session_start_time
    avg_episode_time = sum(episode_times) / len(episode_times) if episode_times else 0
    fps_actual = total_frames / total_time if total_time > 0 else 0

    # Build final report
    # Summary table
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Episodes", str(dataset.meta.total_episodes))
    summary.add_row("Total frames", str(total_frames))
    summary.add_row("Location", output_dir)
    summary.add_row("Conversion config", conversion_config_dest)

    # Per-episode table
    ep_table = Table(title="Per-Episode Breakdown", title_style="bold", title_justify="left", padding=(0, 1))
    ep_table.add_column("#", justify="right", style="dim")
    ep_table.add_column("MCAP File")
    ep_table.add_column("Frames", justify="right")
    ep_table.add_column("Duration", justify="right")
    ep_table.add_column("Speed", justify="right")
    for j, i in enumerate(episode_original_indices):
        mcap_path = mcap_files[i]
        ep_fps = episode_frame_counts[j] / episode_times[j] if episode_times[j] > 0 else 0
        ep_table.add_row(
            str(i + 1),
            mcap_path.name,
            str(episode_frame_counts[j]),
            format_duration(episode_times[j]),
            f"{ep_fps:.1f} f/s",
        )

    # Timing table
    timing = Table(show_header=False, box=None, padding=(0, 2))
    timing.add_column(style="bold")
    timing.add_column()
    timing.add_row("Total time", format_duration(total_time))
    timing.add_row("Avg per episode", format_duration(avg_episode_time))
    timing.add_row("Processing rate", f"{fps_actual:.1f} frames/sec")

    report = Panel(
        Group(summary, "", Padding(ep_table, (0, 0, 0, 2)), "", timing),
        title="[bold green]LeRobot Dataset Created Successfully",
        border_style="green",
        padding=(1, 2),
    )
    console.print(report)

    return dataset


def build_parser(profile: str = "standard") -> argparse.ArgumentParser:
    defaults = _PROFILE_DEFAULTS[profile]
    parser = argparse.ArgumentParser(
        description=defaults["description"],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=defaults["examples"],
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, required=True,
        help="input directory containing MCAP files",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="data/datasets",
        help="output base directory — dataset is saved to <output-dir>/<input-dir-name>/ (default: data/datasets)",
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="full output path override — use this exact directory instead of <output-dir>/<input-dir-name>/",
    )
    parser.add_argument(
        "--config", type=str,
        help="path to YAML config file",
    )
    parser.add_argument(
        "--hf-user", type=str,
        help="Hugging Face username (default: auto-detect)",
    )
    parser.add_argument(
        "--hf-repo", type=str,
        help="dataset repository name (default: output dir name)",
    )
    parser.add_argument(
        "--robot-type", type=str, default="anvil_openarm",
        choices=["anvil_openarm", "anvil_yam"],
        help="robot type (default: anvil_openarm)",
    )
    parser.add_argument(
        "--fps", type=int, default=None,
        help="output fps — overrides auto-detected source fps; must not exceed source fps",
    )
    parser.add_argument(
        "--tolerance-s", type=float, default=1e-3,
        help="timestamp sync tolerance in seconds (default: 0.001)",
    )
    parser.add_argument(
        "--task", type=str, default="manipulation",
        help="task name for the dataset (default: manipulation)",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="upload to Hugging Face Hub after conversion",
    )
    parser.add_argument(
        "--buffer-seconds", type=float, default=5.0,
        help="buffer window for time alignment in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--vcodec", type=str, default=defaults["default_vcodec"],
        choices=["h264", "hevc", "libsvtav1", "auto"],
        help=(
            f"video codec (default: {defaults['default_vcodec']}). "
            "Use 'auto' to let LeRobot pick the best available hardware encoder"
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume conversion — skip already-converted episodes and append new ones",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        metavar="N",
        help="only convert the first N episodes (default: convert all)",
    )
    parser.add_argument(
        "--extra-episode-keys", default="", metavar="K1,K2",
        help="Additional keys to lift from each episode's recorder metadata.json into "
             "meta/episodes as columns. The envelope keys written by label-session "
             "(envelope_size, envelope_facing_side, destination_basket_side, arm) are "
             "picked up automatically when present.",
    )
    parser.add_argument(
        "--act-from-obs-n-step", type=int, default=None,
        metavar="N",
        help="override action_from_observation_n in config: action[t] = observation[t+N] (default: use config value, factory default 10)",
    )
    parser.add_argument(
        "--debug-plot-episodes", type=int, default=defaults["default_debug_plot_episodes"],
        metavar="N",
        help=f"number of episodes to include in debug plots (default: {defaults['default_debug_plot_episodes']})",
    )
    parser.add_argument(
        "--parallel-episodes",
        type=int,
        default=None,
        metavar="N",
        help=(
            "number of episode shards to convert in parallel "
            "(0 = auto, 1 = off, default: profile-specific)"
        ),
    )
    parser.add_argument(
        "--quality-report", type=str, default=None,
        help=(
            "path to a mcap-valid JSON report. A report is REQUIRED to run mcap-convert — "
            "if omitted, it is auto-discovered at <input-dir>/mcap_valid_reports/report.json "
            "(run `mcap-valid -i INPUT_DIR` first to generate it); if neither is found, "
            "mcap-convert exits with an error before touching the output directory"
        ),
    )
    parser.add_argument(
        "--include-flagged",
        choices=_SEVERITY_ORDER,
        default=SEVERITY_WARNING,
        help=(
            "highest severity tier to include when converting, per the quality "
            "report. Inclusive threshold: 'pass' converts only clean episodes "
            "(skips warning AND critical); 'warning' (default) also converts "
            "warning-level episodes, skipping only critical ones automatically; "
            "'critical' converts every episode regardless of severity, skipping "
            "nothing."
        ),
    )
    parser.add_argument(
        "--skip-episode-idx", type=str, default=None,
        help=(
            "manually skip specific episodes by 1-based index, independent of "
            "--quality-report. Accepts a comma-separated list (1,2,5,6), a "
            "colon range with an EXCLUSIVE end matching Python slice convention "
            "(1:4 selects episodes 1,2,3 — NOT 4), an open-ended range (2: or :4), "
            "or a mix (1,3:5,8). Whitespace is tolerated."
        ),
    )
    return parser


def main_with_profile(args=None, profile: str = "standard"):
    """Run the converter with profile-specific defaults."""
    defaults = _PROFILE_DEFAULTS[profile]
    parser = build_parser(profile=profile)
    args = parser.parse_args(args)
    requested_parallel_workers = (
        defaults["parallel_episode_workers"]
        if args.parallel_episodes is None
        else args.parallel_episodes
    )

    # Resolve output path: --output-path wins; otherwise <output-dir>/<input-dir-name>/
    input_name = Path(args.input_dir.rstrip("/")).name
    if args.output_path:
        args.output_dir = args.output_path.rstrip("/")
    else:
        args.output_dir = str(Path(args.output_dir.rstrip("/")) / input_name)

    # Handle HuggingFace username
    if args.hf_user:
        hf_username = args.hf_user
    else:
        try:
            user_info = huggingface_hub.whoami()
            hf_username = user_info["name"]
        except Exception as e:
            log(f"[yellow]Cannot get Hugging Face user info: {e}[/yellow]")
            hf_username = "anvil_robot"

    # Construct repo_id
    dataset_name = args.hf_repo if args.hf_repo else Path(args.output_dir).name
    repo_id = f"{hf_username}/{dataset_name}"

    # Load configuration
    if args.config:
        config = ConfigLoader.from_yaml(args.config)
        log(f"Loaded config from: [dim]{args.config}[/dim]")
    else:
        config = ConfigLoader.get_default()
        log("Using default configuration")

    # ── Mandatory quality-report gate ──────────────────────────────────
    # mcap-convert refuses to run without a mcap-valid quality report (explicit
    # --quality-report, or auto-discovered at the default path) so bad
    # recordings are caught before they enter a dataset. This only checks that
    # a report FILE exists — --include-flagged (below) is a separate mechanism
    # that reads the report's *contents* and defaults to "warning", so only
    # critical episodes are skipped automatically; pass --include-flagged
    # critical to opt out entirely and convert everything.
    report_path = args.quality_report
    default_json, _ = default_report_paths(Path(args.input_dir))
    if report_path is None:
        if default_json.is_file():
            report_path = str(default_json)
    if report_path is None or not Path(report_path).is_file():
        # escape(): input-dir/report paths are user/data-controlled and could
        # otherwise be parsed as Rich markup (e.g. a path containing "[red]").
        console.print(
            "\n[bold red]ERROR: No mcap-valid quality report found for this input.[/bold red]\n"
            "mcap-convert requires a quality report to exist before conversion, so bad\n"
            "recordings are caught before they enter a dataset.\n"
            f"Run mcap-valid first:\n"
            f"  [bold]uv run mcap-valid -i {escape(args.input_dir)}[/bold]\n"
            f"then re-run this command — the report is auto-discovered at\n"
            f"  {escape(str(default_json))}\n"
            "or pass --quality-report PATH to point at a report elsewhere.\n"
        )
        exit(1)

    quality_skip_paths = resolve_quality_skip_paths(report_path, args.include_flagged)

    if args.act_from_obs_n_step is not None:
        config.action_from_observation_n = args.act_from_obs_n_step
        log(f"action_from_observation_n overridden to [bold]{args.act_from_obs_n_step}[/bold] via --act-from-obs-n-step")

    # Collect MCAP files once (reused for fps detection and conversion)
    all_mcap_files = collect_mcap_files(args.input_dir)

    # Validate --skip-episode-idx early (before any output-dir mutation below)
    skip_episode_indices = None
    if args.skip_episode_idx:
        try:
            skip_episode_indices = parse_episode_index_spec(args.skip_episode_idx, len(all_mcap_files))
        except ValueError as exc:
            console.print(f"[red]✗ --skip-episode-idx error: {exc}[/red]")
            exit(1)
        log(f"Manually skipping {len(skip_episode_indices)} episode(s) by index: {sorted(skip_episode_indices)}")

    # Always auto-detect input fps from all episodes (fast — reads MCAP summary only)
    ref_topic = list(config.camera_topic_mapping.keys())[0] if config.camera_topic_mapping else None
    ep_fps_raw = []
    if ref_topic:
        for f in all_mcap_files:
            v = McapReader(str(f)).estimate_fps(ref_topic)
            if v:
                ep_fps_raw.append(v)

    if ep_fps_raw:
        snapped = [snap_fps(v) for v in ep_fps_raw]
        input_fps = snap_fps(min(ep_fps_raw))
        input_fps_label = str(input_fps)
        if len(set(snapped)) > 1:
            input_fps_label = f"{input_fps} [yellow](mixed: {snapped})[/yellow]"
    else:
        input_fps = None
        input_fps_label = "unknown"

    # Resolve output fps: CLI --fps > auto-detect min > 30
    if args.fps is not None:
        fps = args.fps
        output_fps_label = f"{fps} (manual override)"
        if input_fps is not None and fps > input_fps:
            console.print(
                f"\n[bold red]ERROR: Output fps ({fps}) is higher than source session fps ({input_fps}).[/bold red]\n"
                "Upsampling is not supported — it creates duplicate frames and degrades dataset quality.\n"
                f"Use [bold]--fps {input_fps}[/bold] or lower, or omit --fps to use the source fps automatically.\n"
            )
            exit(1)
    elif input_fps is not None:
        fps = input_fps
        output_fps_label = f"{fps} (default as source)"
    else:
        fps = 30
        output_fps_label = "30 (default)"
        log("[yellow]Cannot detect fps — defaulting to 30[/yellow]")

    # Startup banner
    banner = Table(show_header=False, box=None, padding=(0, 2))
    banner.add_column(style="bold")
    banner.add_column()
    banner.add_row("Input directory", args.input_dir)
    banner.add_row("Output directory", args.output_dir)
    banner.add_row("HuggingFace Repo", repo_id)
    banner.add_row("Robot Type", args.robot_type)
    banner.add_row("Source Session FPS", input_fps_label)
    banner.add_row("Output FPS", output_fps_label)
    banner.add_row("Buffer", f"{args.buffer_seconds}s")
    banner.add_row("Profile", defaults["label"])
    banner.add_row("Video codec", args.vcodec)
    banner.add_row("Parallel episodes", "auto" if requested_parallel_workers == 0 else str(requested_parallel_workers))
    banner.add_row("Resume", "yes" if args.resume else "no")
    banner.add_row("Max episodes", str(args.max_episodes) if args.max_episodes else "all")
    if config.action_from_observation:
        n_label = str(config.action_from_observation_n)
        if args.act_from_obs_n_step is not None:
            n_label += " [yellow](CLI override)[/yellow]"
        banner.add_row("act-from-obs n", n_label)
    banner.add_row("Debug plots", f"first {args.debug_plot_episodes} episodes")

    console.print(Panel(
        banner,
        title=defaults["title"],
        border_style="blue",
        padding=(1, 2),
    ))

    try:
        # Determine resume_from: number of already-converted episodes to skip
        resume_from = 0
        if args.resume and os.path.exists(args.output_dir):
            info_path = os.path.join(args.output_dir, "meta", "info.json")
            try:
                with open(info_path) as f:
                    resume_from = json.load(f).get("total_episodes", 0)
                log(f"Resuming from episode [bold]{resume_from}[/bold] — skipping already-converted episodes")
            except Exception as e:
                log(f"[yellow]Cannot read existing metadata ({e}) — starting fresh[/yellow]")
                shutil.rmtree(args.output_dir)
        elif os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
            log("Removed existing output directory")

        # Convert session
        log("[bold]Starting conversion...[/bold]")

        dataset = convert_session(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            repo_id=repo_id,
            robot_type=args.robot_type,
            fps=fps,
            tolerance_s=args.tolerance_s,
            task=args.task,
            config=config,
            buffer_seconds=args.buffer_seconds,
            config_path=args.config,
            vcodec=args.vcodec,
            resume_from=resume_from,
            max_episodes=args.max_episodes,
            mcap_files=all_mcap_files,
            debug_plot_episodes=args.debug_plot_episodes,
            quality_skip_paths=quality_skip_paths,
            skip_episode_indices=skip_episode_indices,
            streaming_encoding=defaults["streaming_encoding"],
            encoder_queue_maxsize=defaults["encoder_queue_maxsize"],
            encoder_threads=defaults["encoder_threads"],
            progress_update_every=defaults["progress_update_every"],
            use_live_progress=console.is_terminal and sys.stdout.isatty(),
            parallel_episode_workers=requested_parallel_workers,
            extra_episode_keys=tuple(
                k.strip() for k in args.extra_episode_keys.split(",") if k.strip()
            ),
        )

        # Upload to Hub if requested
        if args.push_to_hub:
            with console.status("[bold]Uploading dataset to Hugging Face Hub..."):
                dataset.push_to_hub()
            log("[green]Dataset uploaded successfully![/green]")

    except Exception:
        console.print_exception()
        exit(1)


def main(args=None):
    """Main entry point for the standard converter path."""
    return main_with_profile(args=args, profile="standard")


if __name__ == "__main__":
    main()
