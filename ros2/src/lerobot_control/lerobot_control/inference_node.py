#!/usr/bin/env python3
"""
LeRobot Inference Node for Robot Arms

Multi-process inference node with shared-memory image workers.

Usage:
    ros2 run lerobot_control inference_node \
        --ros-args -p model_path:=/path/to/model -p config_file:=/path/to/config.yaml

Subscribes to:
    - Joint states topic (sensor_msgs/JointState)
    - Camera image topics (sensor_msgs/CompressedImage)

Publishes:
    - Forward position controller command topics (std_msgs/Float64MultiArray)
    - /monitor/obs_state, /monitor/raw_output, /monitor/control_cmd  (when monitor_enable:=true)
"""

import hashlib
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
import torch
import yaml
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from .action_limiter import ActionLimiter
from .delta_restore import resolve_action_type, restore_delta_chunk
from .input_watchdog import (
    InputSnapshot,
    InputWatchdog,
    ObservationProvenance,
    ObservationSequence,
    WatchdogResult,
)
from .metrics_tracker import MetricsTracker
from .model_loader import ModelLoader, set_deterministic_mode


@dataclass(frozen=True)
class RTCReadinessAssessment:
    """Exact queue, age, and guidance margins for one RTC refill."""

    sustainable: bool
    candidate_delay_steps: int
    q_start: int
    wait_steps: int
    q_trigger: int
    scheduler_guard_steps: int
    q_required: int
    latency_bound_sec: float
    refill_delay_bound_steps: int
    coverage_required_steps: int
    age_at_next_refill_sec: float
    useful_guided_overlap_steps: int
    failures: tuple[str, ...]


@dataclass(frozen=True)
class RTCMergeAlignment:
    """Observed queue consumption and the delay used to align a new chunk."""

    requested_at_monotonic: float
    merge_at_monotonic: float
    runtime_sec: float
    wall_delay_steps: int
    consumed_steps: int
    merge_delay_steps: int
    queue_size_at_request: int
    queue_size_at_merge: int


@dataclass(frozen=True)
class RTCDispatchSnapshot:
    """Identity, depth, index, and clock captured with an RTC leftover."""

    queue: object
    queue_size: int
    action_index: int
    requested_at_monotonic: float


@dataclass(frozen=True)
class VLAObservationTiming:
    """Monotonic stages used to build one preprocessed VLA observation."""

    callback_started_monotonic: float
    read_started_monotonic: float
    read_completed_monotonic: float
    preprocess_started_monotonic: float
    ready_at_monotonic: float


@dataclass(frozen=True)
class PendingRTCCudaTiming:
    """Non-blocking CUDA events correlated with one RTC timing sample."""

    sample_id: int
    start_event: object
    end_event: object


@dataclass
class RTCMergeStageTiming:
    """Optional diagnostic timestamps around the action-queue merge."""

    queue_lock_requested_at_monotonic: float | None = None
    queue_lock_acquired_at_monotonic: float | None = None
    alignment_merge_at_monotonic: float | None = None
    merge_completed_at_monotonic: float | None = None


class LeRobotInferenceNode(Node):
    """
    ROS2 node for LeRobot model inference and robot control.

    Uses multi-process strategy with shared-memory image workers for
    GIL-free JPEG decompression and true parallel camera processing.
    """

    def __init__(self, parameter_overrides: list = None):
        super().__init__("lerobot_inference_node", parameter_overrides=parameter_overrides or [])

        self._subscription_callback_group = ReentrantCallbackGroup()

        self._setup_config()

        self.metrics = MetricsTracker()
        self.strategy = self._create_strategy()
        self.strategy.setup(
            node=self,
            config={"device": self.device, **self.config},
            camera_mapping=self.camera_mapping,
            joint_names_config=self.joint_names_config,
            joint_state_topic=self.joint_state_topic,
            image_shape=self.image_shape,
            metrics=self.metrics,
            callback_group=self._subscription_callback_group,
            debug_image_dir=self._debug_image_dir,
        )

        # The safety lock is always the outermost lock when locks are nested.
        # The model, observation, and action-queue locks are never nested with
        # one another. Inference never enters a queue unless both its captured
        # watchdog epoch and policy epoch are still current.
        self._safety_lock = threading.RLock()
        self._model_lock = threading.Lock()
        self._action_queue_lock = threading.Lock()
        self._policy_epoch: int = 0
        self._watchdog = InputWatchdog(
            camera_timeout_sec=self._watchdog_camera_timeout_sec,
            joint_state_timeout_sec=self._watchdog_joint_timeout_sec,
            max_sensor_skew_sec=self._watchdog_max_sensor_skew_sec,
            max_action_age_sec=self._watchdog_max_action_age_sec,
            startup_grace_sec=self._watchdog_startup_grace_sec,
            started_at_monotonic=time.monotonic(),
        )

        # Non-VLA action buffer (ACT/Diffusion put actions here from obs timer)
        self._classic_action_deque: deque = deque(maxlen=10)
        self._classic_chunk_source_monotonic: float | None = None
        self._vla_action_source_monotonic: float | None = None
        self._vla_action_epoch: int | None = None
        # Reference joint state captured at the moment each action chunk was
        # generated (in model/observation order).  All queued steps in the chunk
        # share this reference so delta restoration is consistent with training.
        # _delta_ref_state and _abs_shadow_queue must always be reset together;
        # use _reset_delta_state() in any future reload or episode-boundary path.
        self._delta_ref_state: np.ndarray | None = None
        self._abs_shadow_queue: deque[np.ndarray] = deque()

        self._shutting_down: bool = False
        self._has_published: bool = False

        if not self.echo_topic_only:
            self._setup_model()

            self.action_limiter = ActionLimiter(
                max_delta=self.max_position_delta,
                min_delta_threshold=self.min_position_delta,
                model_joint_order=self.joint_names_config.get("model_joint_order", []),
                controller_joint_order=self.joint_names_config.get("controller_joint_order", []),
                logger=self.get_logger(),
            )

            # Resolve delta exclude indices in model joint order (used by chunk restore)
            _model_order = self.joint_names_config.get("model_joint_order", [])
            self._delta_exclude_indices = [
                _model_order.index(name)
                for name in self.delta_exclude_joints
                if name in _model_order
            ]

            self._setup_publishers()
            self._watchdog_service_group = MutuallyExclusiveCallbackGroup()
            self._watchdog_rearm_service = self.create_service(
                Trigger,
                "~/rearm_watchdog",
                self._handle_watchdog_rearm,
                callback_group=self._watchdog_service_group,
            )

            # Unified split-timer architecture for all models:
            #   _obs_update:    preprocess (+ inference for non-VLA)
            #   _publish_loop:  pop action from queue/deque → publish
            self._obs_callback_group = MutuallyExclusiveCallbackGroup()
            self._publish_callback_group = MutuallyExclusiveCallbackGroup()

            self._obs_timer = self.create_timer(
                1.0 / self.control_freq,
                self._obs_update,
                callback_group=self._obs_callback_group,
            )
            self._publish_timer = self.create_timer(
                1.0 / self.control_freq,
                self._publish_loop,
                callback_group=self._publish_callback_group,
            )
            # Model loading can take long enough to exceed the configured grace
            # period. Start the safety clock and RTC worker only after publishers,
            # queues, and callbacks are fully initialized.
            self._watchdog.started_at_monotonic = time.monotonic()
            if self._is_vla:
                self._start_inference_thread()

        self._log_startup()

        # Debug mode: enables ActionSmoothTracker, queue depth stats, Action FPS
        self._smooth_tracker = None
        self._queue_depths: deque[int] = deque(maxlen=300)
        self._vla_skip_count: int = 0
        if self._debug and not self.echo_topic_only and hasattr(self, "model"):
            from .action_smooth_tracker import ActionSmoothTracker

            total_action_dim = sum(
                ac.get("action_end", 0) - ac.get("action_start", 0)
                for ac in self.arms_config.values()
            )
            if total_action_dim > 0:
                self._smooth_tracker = ActionSmoothTracker(action_dim=total_action_dim)

        # Stats logging timer (in publish callback group to avoid race on _queue_depths)
        self._stats_log_interval = 5.0
        self._stats_timer = self.create_timer(
            self._stats_log_interval,
            self._log_input_stats,
            callback_group=self._publish_callback_group if not self.echo_topic_only else MutuallyExclusiveCallbackGroup(),
        )

        # Windowed rate tracking
        self._prev_log_time: float | None = None
        self._prev_joint_count: int = 0
        self._prev_control_count: int = 0
        self._prev_inference_count: int = 0
        self._prev_action_output_count: int = 0
        self._prev_frame_counters: dict[str, int] = {}

    def _setup_config(self) -> None:
        """Declare ROS2 params, load YAML, and read all checkpoint metadata."""
        self.declare_parameter("model_path", "")
        self.declare_parameter("config_file", "")
        self.declare_parameter("control_frequency", 30.0)
        self.declare_parameter("enforce_joint_position_limits", True)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("deterministic", False)
        self.declare_parameter("deterministic_seed", 42)
        self.declare_parameter("echo_topic_only", False)
        self.declare_parameter("debug", False)
        self.declare_parameter("debug_image_dir", "")
        self.declare_parameter("monitor_enable", False)
        self.declare_parameter("joint_state_worker", False)

        # Static fields from ROS2 params
        self.echo_topic_only = self.get_parameter("echo_topic_only").value
        self._debug = self.get_parameter("debug").value
        self._monitor_enable: bool = self.get_parameter("monitor_enable").value
        _debug_image_dir = self.get_parameter("debug_image_dir").value
        self._debug_image_dir: str | None = _debug_image_dir if _debug_image_dir else None
        self.model_path = self.get_parameter("model_path").value
        if not self.model_path and not self.echo_topic_only:
            raise ValueError("model_path parameter is required")

        self.control_freq = self.get_parameter("control_frequency").value
        self._enforce_joint_position_limits = self.get_parameter(
            "enforce_joint_position_limits"
        ).value
        if not isinstance(self._enforce_joint_position_limits, bool):
            raise ValueError(
                "enforce_joint_position_limits parameter must be a boolean"
            )
        self.device = self.get_parameter("device").value
        if not math.isfinite(self.control_freq) or self.control_freq <= 0:
            raise ValueError("control_frequency must be finite and > 0")

        # Load YAML config
        config_file = self.get_parameter("config_file").value
        self.config = self._load_yaml_config(config_file)
        joint_state_worker = self.get_parameter("joint_state_worker").value
        if not isinstance(joint_state_worker, bool):
            raise ValueError("joint_state_worker parameter must be a boolean")
        if joint_state_worker:
            runtime_config = self.config.setdefault("runtime", {})
            if not isinstance(runtime_config, dict):
                raise ValueError("runtime must be a mapping")
            runtime_config["joint_state_worker"] = True

        diagnostics_config = self.config.get("diagnostics", {})
        self._rtc_timing_enabled = diagnostics_config.get("rtc_timing", False)
        self._rtc_cuda_timing_enabled = diagnostics_config.get(
            "rtc_cuda_events", False
        )
        self._rtc_provenance_enabled = diagnostics_config.get(
            "rtc_provenance", False
        )
        if not isinstance(self._rtc_timing_enabled, bool):
            raise ValueError("diagnostics.rtc_timing must be a boolean")
        if not isinstance(self._rtc_cuda_timing_enabled, bool):
            raise ValueError("diagnostics.rtc_cuda_events must be a boolean")
        if not isinstance(self._rtc_provenance_enabled, bool):
            raise ValueError("diagnostics.rtc_provenance must be a boolean")
        if self._rtc_cuda_timing_enabled and not self._rtc_timing_enabled:
            raise ValueError(
                "diagnostics.rtc_cuda_events requires diagnostics.rtc_timing"
            )

        # Fields from YAML config
        safety_config = self.config.get("safety", {})
        self.max_position_delta = safety_config.get("max_position_delta", 0.1)
        self.min_position_delta = safety_config.get("min_position_delta", None)
        if (
            not math.isfinite(self.max_position_delta)
            or self.max_position_delta <= 0
        ):
            raise ValueError("safety.max_position_delta must be finite and > 0")
        if self.min_position_delta is not None and (
            not math.isfinite(self.min_position_delta)
            or self.min_position_delta < 0
        ):
            raise ValueError("safety.min_position_delta must be finite and >= 0")
        self._joint_limit_tolerance = self._parse_joint_limit_tolerance(
            safety_config.get("joint_limit_tolerance", 1e-6)
        )
        raw_joint_position_limits = safety_config.get("joint_position_limits", {})
        raw_saturate_joint_targets = safety_config.get("saturate_joint_targets", [])
        raw_saturate_joint_margins = safety_config.get("saturate_joint_margins", {})

        watchdog_config = self.config.get("watchdog", {})
        self._watchdog_camera_timeout_sec = float(
            watchdog_config.get("camera_timeout_sec", 0.25)
        )
        self._watchdog_joint_timeout_sec = float(
            watchdog_config.get("joint_state_timeout_sec", 0.10)
        )
        self._watchdog_startup_grace_sec = float(
            watchdog_config.get("startup_grace_sec", 10.0)
        )
        self._watchdog_max_sensor_skew_sec = float(
            watchdog_config.get("max_sensor_skew_sec", 0.10)
        )
        self._watchdog_max_action_age_sec = float(
            watchdog_config.get("max_action_age_sec", 1.50)
        )
        if (
            not math.isfinite(self._watchdog_camera_timeout_sec)
            or self._watchdog_camera_timeout_sec <= 0
        ):
            raise ValueError("watchdog.camera_timeout_sec must be finite and > 0")
        if (
            not math.isfinite(self._watchdog_joint_timeout_sec)
            or self._watchdog_joint_timeout_sec <= 0
        ):
            raise ValueError("watchdog.joint_state_timeout_sec must be finite and > 0")
        if (
            not math.isfinite(self._watchdog_startup_grace_sec)
            or self._watchdog_startup_grace_sec < 0
        ):
            raise ValueError("watchdog.startup_grace_sec must be finite and >= 0")
        if self._watchdog_max_sensor_skew_sec <= 0:
            raise ValueError("watchdog.max_sensor_skew_sec must be finite and > 0")
        if not math.isfinite(self._watchdog_max_sensor_skew_sec):
            raise ValueError("watchdog.max_sensor_skew_sec must be finite and > 0")
        if (
            not math.isfinite(self._watchdog_max_action_age_sec)
            or self._watchdog_max_action_age_sec <= 0
        ):
            raise ValueError("watchdog.max_action_age_sec must be finite and > 0")

        self.joint_state_topic = self.config.get("joint_state_topic", "/joint_states")
        _cameras_cfg: dict = self.config.get("cameras", {})
        self.camera_mapping = _cameras_cfg.get("mapping", {})
        self.camera_names = list(self.camera_mapping.values())
        duplicate_camera_names = sorted(
            name for name in set(self.camera_names) if self.camera_names.count(name) > 1
        )
        if duplicate_camera_names:
            raise ValueError(
                "camera mappings must use unique model feature names; duplicates: "
                + ", ".join(duplicate_camera_names)
            )

        # Build per-camera expected fps dict (camera name → expected fps).
        # Warning threshold = expected * 2/3, independent of control_frequency.
        _global_expected_fps: float = _cameras_cfg.get("fps", 30.0)
        _fps_overrides: dict = _cameras_cfg.get("fps_overrides", {})
        # overrides are keyed by ROS topic; map to camera name via camera_mapping
        self._expected_camera_fps: dict[str, float] = {
            name: _fps_overrides.get(topic, _global_expected_fps)
            for topic, name in self.camera_mapping.items()
        }

        self.arms_config = self.config.get("arms", {})
        self.joint_names_config = self.config.get("joint_names", {})
        if raw_joint_position_limits or not self.echo_topic_only:
            self._joint_position_limits = self._parse_joint_position_limits(
                raw_joint_position_limits
            )
        else:
            self._joint_position_limits = {}
        self._saturate_joint_targets = self._parse_saturate_joint_targets(
            raw_saturate_joint_targets
        )
        self._saturate_joint_margins = self._parse_saturate_joint_margins(
            raw_saturate_joint_margins
        )
        self._saturation_counts: dict[str, int] = {}

        # Inference tuning — per model type (resolved after model_type is known)
        self._tuning_config = self.config.get("inference_tuning", {})

        # --- Checkpoint metadata (lightweight JSON reads, no tensor loading) ---
        # Skip in echo_topic_only mode — no checkpoint needed
        meta = {} if self.echo_topic_only else self._read_checkpoint_metadata()

        # image_shape: from config.json input_features — must match training
        # Default (480, 640, 3) is used only in echo_topic_only mode with no checkpoint
        self.image_shape = meta.get("image_shape", (480, 640, 3))

        # model_type: from config.json, YAML overrides if explicitly set
        model_cfg = self.config.get("model", {})
        self.model_type = model_cfg.get("type") or meta.get("model_type")

        # action_type from anvil_config.json — must match training
        self.action_type: str = resolve_action_type(meta)
        self.use_delta_actions: bool = self.action_type in ("delta_obs_t", "delta_sequential")
        self.delta_exclude_joints: list[str] = meta.get("delta_exclude_joints", [])

        # Resolve delta exclude joint indices (in model output order)
        # Will be finalized after joint_names_config is loaded.
        self._delta_exclude_indices: list[int] = []

        # task_description: anvil_config.json first, YAML overrides if explicitly set
        self.task_description = meta.get("task_description", "")
        if model_cfg.get("task_description"):
            self.task_description = model_cfg["task_description"]


    @property
    def _is_vla(self) -> bool:
        """True if the loaded model is a VLA (pi0 / pi05 / smolvla)."""
        return getattr(self, "model_type", None) in {"smolvla", "pi0", "pi05"}

    def _load_yaml_config(self, config_file: str) -> dict:
        """Load configuration from YAML file."""
        if not config_file:
            self.get_logger().warn("No config_file specified, using defaults")
            return {}

        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_path) as f:
            return yaml.safe_load(f)

    def _expected_controller_joint_names(self) -> tuple[str, ...]:
        """Return every full ROS joint name targeted by configured action slices."""
        controller_order = self.joint_names_config.get(
            "controller_joint_order",
            self.joint_names_config.get("joint_order", []),
        )
        names: list[str] = []
        for arm_name, arm_config in self.arms_config.items():
            start = int(arm_config.get("action_start", 0))
            end = int(arm_config.get("action_end", 0))
            arm_dim = end - start
            ros_prefix = arm_config.get("ros_prefix", arm_name)
            if arm_dim <= 0 or len(controller_order) < arm_dim:
                raise ValueError(
                    f"invalid action slice/controller order for arm {arm_name}: "
                    f"slice=[{start}:{end}], controller joints={len(controller_order)}"
                )
            names.extend(f"{ros_prefix}_{joint}" for joint in controller_order[:arm_dim])
        if len(names) != len(set(names)):
            raise ValueError("configured controller joint names are not unique")
        return tuple(names)

    @staticmethod
    def _parse_joint_limit_tolerance(raw_tolerance: float) -> float:
        """Validate the small numerical tolerance used for absolute limits."""
        tolerance = float(raw_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0 or tolerance > 1e-6:
            raise ValueError(
                "safety.joint_limit_tolerance must be finite and between 0 and 1e-6"
            )
        return tolerance

    def _parse_saturate_joint_targets(self, raw_names) -> frozenset[str]:
        """Validate joints allowed to clamp to their own absolute limits."""
        if not raw_names:
            return frozenset()
        if isinstance(raw_names, str) or not isinstance(raw_names, (list, tuple)):
            raise ValueError("safety.saturate_joint_targets must be a list of names")

        names = frozenset(str(name) for name in raw_names)
        unknown = sorted(names - set(self._joint_position_limits))
        if unknown:
            raise ValueError(
                "safety.saturate_joint_targets must reference joints declared in "
                "safety.joint_position_limits: unknown=" + ",".join(unknown)
            )
        return names

    # This is an acceptance band for a bounded recording artefact, never extra
    # robot travel. The published target remains clamped to the hard limit.
    MAX_SATURATION_MARGIN_RAD = 0.05

    def _parse_saturate_joint_margins(self, raw_margins) -> dict[str, float]:
        """Validate the bounded overshoot accepted for each saturating joint."""
        if not self._saturate_joint_targets:
            if raw_margins:
                raise ValueError(
                    "safety.saturate_joint_margins is set but "
                    "safety.saturate_joint_targets is empty"
                )
            return {}
        if not isinstance(raw_margins, dict):
            raise ValueError("safety.saturate_joint_margins must be a mapping")

        actual = set(raw_margins)
        missing = sorted(self._saturate_joint_targets - actual)
        extra = sorted(actual - self._saturate_joint_targets)
        if missing or extra:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if extra:
                details.append("unexpected=" + ",".join(extra))
            raise ValueError(
                "safety.saturate_joint_margins must exactly cover "
                "safety.saturate_joint_targets: " + "; ".join(details)
            )

        parsed: dict[str, float] = {}
        for name in sorted(self._saturate_joint_targets):
            margin = float(raw_margins[name])
            if not math.isfinite(margin) or margin <= 0.0:
                raise ValueError(
                    f"safety.saturate_joint_margins[{name}] must be positive "
                    f"and finite, got {margin}"
                )
            if margin > self.MAX_SATURATION_MARGIN_RAD:
                raise ValueError(
                    f"safety.saturate_joint_margins[{name}]={margin} exceeds "
                    f"the {self.MAX_SATURATION_MARGIN_RAD} rad ceiling"
                )
            parsed[name] = margin
        return parsed

    def _parse_joint_position_limits(
        self,
        raw_limits: dict,
    ) -> dict[str, tuple[float, float]]:
        """Validate a complete absolute limit mapping for every commanded joint."""
        expected_names = self._expected_controller_joint_names()
        if not raw_limits:
            raise ValueError(
                "safety.joint_position_limits is required for real-robot inference"
            )
        if not isinstance(raw_limits, dict):
            raise ValueError("safety.joint_position_limits must be a mapping")

        actual_names = set(raw_limits)
        missing = sorted(set(expected_names) - actual_names)
        extra = sorted(actual_names - set(expected_names))
        if missing or extra:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if extra:
                details.append("unexpected=" + ",".join(extra))
            raise ValueError(
                "safety.joint_position_limits must exactly cover commanded joints: "
                + "; ".join(details)
            )

        parsed: dict[str, tuple[float, float]] = {}
        for name in expected_names:
            bounds = raw_limits[name]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"joint limit for {name} must be [lower, upper]")
            lower, upper = (float(bounds[0]), float(bounds[1]))
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(
                    f"invalid joint limit for {name}: lower={lower}, upper={upper}"
                )
            parsed[name] = (lower, upper)
        return parsed

    def _read_checkpoint_metadata(self) -> dict:
        """
        Read checkpoint metadata from config.json and anvil_config.json.
        Lightweight — JSON only, no tensor loading.
        Raises RuntimeError if model_path is set but config.json is missing/unreadable.
        """
        if not self.model_path:
            return {}

        checkpoint = Path(self.model_path)

        # Auto-detect pretrained_model subdirectory (mirrors ModelLoader logic)
        pretrained = checkpoint / "pretrained_model"
        if pretrained.exists() and (pretrained / "config.json").exists():
            checkpoint = pretrained

        # Auto-detect HF cache snapshot structure (blobs/ + snapshots/)
        if not (checkpoint / "config.json").exists():
            snapshots = checkpoint / "snapshots"
            if snapshots.is_dir():
                for snap in sorted(snapshots.iterdir(), reverse=True):
                    if (snap / "config.json").exists():
                        checkpoint = snap
                        break

        # config.json — required
        config_path = checkpoint / "config.json"
        if not config_path.exists():
            raise RuntimeError(f"config.json not found in {checkpoint}")
        cfg = json.loads(config_path.read_text())

        # image shape from input_features (first VISUAL entry)
        image_shape = None
        for feat in cfg.get("input_features", {}).values():
            if feat.get("type") == "VISUAL":
                c, h, w = feat["shape"]   # stored as [C, H, W]
                image_shape = (h, w, c)   # return as (H, W, C) for cv2
                break
        if image_shape is None:
            raise RuntimeError(f"No VISUAL input feature found in {config_path}")

        # Update model_path to resolved checkpoint (for ModelLoader)
        self.model_path = str(checkpoint)

        meta = {
            "image_shape": image_shape,
            "model_type":  cfg.get("type"),
        }

        # anvil_config.json — optional (absent for checkpoints pre-anvil_config)
        anvil_path = checkpoint / "anvil_config.json"
        if anvil_path.exists():
            anvil = json.loads(anvil_path.read_text())
            meta["action_type"] = anvil.get("action_type", "absolute")
            meta["use_delta_actions"] = anvil.get("use_delta_actions", False)
            meta["delta_exclude_joints"] = anvil.get("delta_exclude_joints", [])
            if "task_description" in anvil:
                meta["task_description"] = anvil["task_description"]
        return meta

    def _create_strategy(self):
        """Create multi-process inference strategy."""
        from .strategies.multi_process import MultiProcessStrategy

        return MultiProcessStrategy()

    def _setup_model(self) -> None:
        """Load model weights and processors. All config fields must be set by _setup_config()."""
        if self.get_parameter("deterministic").value:
            seed = self.get_parameter("deterministic_seed").value
            set_deterministic_mode(seed)
            self.get_logger().info(f"Deterministic mode enabled with seed={seed}")

        # Resolve inference tuning per model type
        tuning = self._tuning_config
        config_overrides = {}

        if self._is_vla:
            self.rtc_config_yaml = tuning.get("rtc", {})
        elif self.model_type == "diffusion":
            diff = tuning.get("diffusion", {})
            if diff.get("n_action_steps") is not None:
                config_overrides["n_action_steps"] = diff["n_action_steps"]
            if diff.get("num_inference_steps") is not None:
                config_overrides["num_inference_steps"] = diff["num_inference_steps"]
        else:  # ACT and others
            act = tuning.get("act", {})
            if act.get("n_action_steps") is not None:
                config_overrides["n_action_steps"] = act["n_action_steps"]
            if act.get("temporal_ensemble_coeff") is not None:
                config_overrides["temporal_ensemble_coeff"] = act["temporal_ensemble_coeff"]
                if act.get("n_action_steps") is None or act["n_action_steps"] > 1:
                    self.get_logger().warn(
                        "temporal_ensemble requires n_action_steps=1, forcing override"
                    )
                    config_overrides["n_action_steps"] = 1

        # Fallback: also check old top-level rtc key for backward compatibility
        if self._is_vla and not self.rtc_config_yaml:
            self.rtc_config_yaml = self.config.get("rtc", {})

        self.n_action_steps_override = config_overrides.get("n_action_steps")

        loader = ModelLoader(
            self.model_path,
            self.device,
            self.model_type,
            config_overrides=config_overrides,
            logger=self.get_logger(),
            rtc_config_yaml=getattr(self, "rtc_config_yaml", {}),
            require_checkpoint_manifest=self.config.get("model", {}).get(
                "require_checkpoint_manifest", True
            ),
        )
        self.model, self.preprocessor, self.postprocessor = loader.load_with_processors()
        self._loader = loader

        # Confirm final model_type (ModelLoader auto-detects if None was passed)
        self.model_type = loader.model_type

        # VLA models: set up ActionQueue and start background inference thread
        if self._is_vla:
            self._setup_vla_inference()
        else:
            # Classic (ACT/Diffusion): initialise latency tracker
            from lerobot_control.latency_stats import LatencyStats

            self._latency_tracker = LatencyStats(maxlen=100)

        if self.model_type in {"smolvla", "pi0", "pi05"} and not self.task_description:
            self.get_logger().warn(
                f"{self.model_type} has no task_description — re-train with --task-description "
                "or set model.task_description in the inference YAML."
            )

    def _log_startup(self) -> None:
        """Log unified startup summary after all setup is complete."""
        logger = self.get_logger()
        logger.info("=" * 50)
        logger.info("LeRobot Inference Node")
        logger.info("=" * 50)
        if self.echo_topic_only:
            logger.info("Mode:       Monitor Only (no model, no publishing)")
        else:
            logger.info(f"Model:      {self.model_path}")
            logger.info(f"Type:       {self.model_type or 'unknown'}")
            logger.info(f"Action type: {self.action_type}")
            if self.use_delta_actions and self.delta_exclude_joints:
                logger.info(f"Delta excl: {self.delta_exclude_joints}")
            if self.model_type in {"smolvla", "pi0", "pi05"}:
                logger.info(f"Task:       '{self.task_description}'")
        logger.info(f"Device:     {self.device}")
        logger.info(f"Frequency:  {self.control_freq} Hz")
        if not self.echo_topic_only:
            logger.info(f"Max delta:  {self.max_position_delta} rad")
            if self._enforce_joint_position_limits:
                logger.info("Joint limits: ENFORCED (configured absolute ranges)")
            else:
                logger.warn(
                    "Joint limits: DISABLED for attended evaluation; commands are "
                    "not clamped or rejected by configured absolute ranges"
                )
            logger.info(
                "Watchdog:  fail-closed "
                f"(camera={self._watchdog_camera_timeout_sec:.3f}s, "
                f"joints={self._watchdog_joint_timeout_sec:.3f}s, "
                f"skew={self._watchdog_max_sensor_skew_sec:.3f}s, "
                f"action_age={self._watchdog_max_action_age_sec:.3f}s, "
                f"startup={self._watchdog_startup_grace_sec:.1f}s)"
            )
            logger.info("Rearm:     ~/rearm_watchdog (std_srvs/Trigger)")

        h, w, _ = self.image_shape
        res_note = "auto-detected from checkpoint" if self.model_path else "default"
        logger.info(f"Resolution: {w}x{h}  ({res_note})")

        logger.info(f"Cameras:    {self.camera_names}")
        logger.info(f"Arms:       {list(self.arms_config.keys())}")

        if not self.echo_topic_only and hasattr(self, "model") and hasattr(self.model, "config"):
            config = self.model.config
            chunk_size = getattr(config, "chunk_size", None)
            n_action_steps = getattr(config, "n_action_steps", None)
            cs = str(chunk_size) if chunk_size is not None else "N/A"
            nas = str(n_action_steps) if n_action_steps is not None else "N/A"

            logger.info("┌─ Inference tuning ──────────────────────────────────────┐")
            logger.info(f"│  chunk_size      = {cs:<4} (fixed at training, read-only)   │")
            logger.info(f"│  n_action_steps  = {nas:<4} (override in inference_tuning:)  │")
            logger.info( "│    → jittery / oscillating?  raise n_action_steps       │")
            logger.info( "│    → hesitates / freezes?    lower n_action_steps       │")
            logger.info( "└─────────────────────────────────────────────────────────┘")

            orig = getattr(self._loader, "checkpoint_n_action_steps", None)
            if (
                orig is not None
                and n_action_steps is not None
                and orig != n_action_steps
                and self.n_action_steps_override is not None
            ):
                logger.info(f"  (overridden from checkpoint default: {orig} → {n_action_steps})")

            if getattr(config, "temporal_ensemble_coeff", None) is not None:
                if hasattr(self.model, "temporal_ensembler"):
                    logger.info("Temporal ensembler initialized successfully")
                else:
                    logger.error("temporal_ensemble_coeff is set but ensembler not created!")

        # GPU/CPU memory after model load
        if not self.echo_topic_only and hasattr(self, "model"):
            if torch.cuda.is_available():
                gpu_mb = torch.cuda.memory_allocated(self.device) / 1e6
                logger.info(f"GPU memory (weights): {gpu_mb:.0f} MB")
            try:
                import psutil

                cpu_mb = psutil.Process().memory_info().rss / 1e6
                logger.info(f"CPU RSS after load:   {cpu_mb:.0f} MB")
            except ImportError:
                pass

        if not self.echo_topic_only and self._is_vla:
            rtc = self.rtc_config_yaml
            logger.info("┌─ RTC ───────────────────────────────────────────────────┐")
            logger.info("│  Status:              ENABLED                           │")
            logger.info(f"│  execution_horizon  = {rtc.get('execution_horizon', 10):<4}                             │")
            logger.info(f"│  max_guidance_weight= {rtc.get('max_guidance_weight', 10.0):<6}                           │")
            logger.info(f"│  attention_schedule = {rtc.get('prefix_attention_schedule', 'EXP'):<6}                           │")
            logger.info(f"│  queue_threshold    = {rtc.get('queue_trigger_threshold', 30):<4}                             │")
            logger.info(
                f"│  readiness_forwards = {rtc.get('readiness_guided_forwards', 5):<4}"
                "                             │"
            )
            logger.info(
                f"│  latency_guard      = {rtc.get('readiness_latency_guard_steps', 2):<4}"
                " steps                       │"
            )
            logger.info(
                "│  index_phase_guard  = "
                f"{rtc.get('readiness_index_phase_tolerance_steps', 1):<4}"
                " steps                       │"
            )
            logger.info(
                "│  scheduler_guard    = "
                f"{rtc.get('readiness_scheduler_guard_steps', 1):<4}"
                " steps                       │"
            )
            logger.info(
                "│  min_guided_overlap = "
                f"{rtc.get('readiness_min_guided_overlap_steps', 3):<4}"
                " steps                       │"
            )
            logger.info("└─────────────────────────────────────────────────────────┘")

    def _setup_publishers(self) -> None:
        """Setup action publishers."""
        self.arm_publishers: dict[str, rclpy.publisher.Publisher] = {}
        for arm_name, arm_config in self.arms_config.items():
            cmd_topic = arm_config.get(
                "command_topic",
                f"/{arm_name}_forward_position_controller/commands",
            )
            self.arm_publishers[arm_name] = self.create_publisher(Float64MultiArray, cmd_topic, 10)
            self.get_logger().info(f"Publishing to: {cmd_topic}")

        if self._monitor_enable:
            self._monitor_obs_pub = self.create_publisher(Float64MultiArray, "/monitor/obs_state", 10)
            self._monitor_raw_pub = self.create_publisher(Float64MultiArray, "/monitor/raw_output", 10)
            self._monitor_cmd_pub = self.create_publisher(Float64MultiArray, "/monitor/control_cmd", 10)
            self.get_logger().info("Monitor topics enabled: /monitor/{obs_state,raw_output,control_cmd}")

    def _new_vla_action_queue(self):
        """Create an empty RTC action queue from the loaded model config."""
        from lerobot.policies.rtc.action_queue import ActionQueue

        return ActionQueue(self.model.config.rtc_config)

    def _invalidate_action_state_locked(self) -> None:
        """Clear observations and every external action buffer.

        The caller must hold ``_safety_lock``. Replacing the RTC queue, rather
        than draining it, prevents a publisher from retaining a reference to an
        old action after a safety or policy epoch transition. Advancing the
        policy epoch also rejects a forward pass that was already in flight;
        policy resets do not alter the input-watchdog state or epoch.
        """
        self._policy_epoch = getattr(self, "_policy_epoch", 0) + 1
        self._classic_action_deque.clear()
        self._classic_chunk_source_monotonic = None
        self._vla_action_source_monotonic = None
        self._vla_action_epoch = None
        self._reset_delta_state()

        # A fault, explicit rearm, or policy reset must return RTC to the same
        # fail-closed startup state. The next forward pass warms the runtime but
        # can never enter the action queue; only a subsequent fresh result may
        # make the policy ready for publication.
        if self._is_vla:
            self._vla_warmup_pending = True
            self._vla_policy_ready = False
            self._vla_stale_result_count = 0
            self._vla_seeded = False
            self._rtc_guided_streak = 0
            if hasattr(self, "_rtc_guided_latencies"):
                self._rtc_guided_latencies.clear()
            if hasattr(self, "_latency_tracker"):
                self._latency_tracker.reset()

        if hasattr(self, "action_limiter") and hasattr(self.action_limiter, "reset"):
            self.action_limiter.reset()

        # Classic policies keep additional normalized action chunks internally.
        # Empty them immediately on a fault without invoking a potentially slow
        # model reset from a timer callback.
        if hasattr(self, "model"):
            with self._model_lock:
                if hasattr(self.model, "_action_queue"):
                    queue = self.model._action_queue
                    if hasattr(queue, "clear"):
                        queue.clear()
                if hasattr(self.model, "_queues") and self.model._queues is not None:
                    queues = (
                        self.model._queues.values()
                        if hasattr(self.model._queues, "values")
                        else (self.model._queues,)
                    )
                    for queue in queues:
                        if hasattr(queue, "clear"):
                            queue.clear()

        if hasattr(self, "_obs_lock"):
            with self._obs_lock:
                self._latest_obs = None
                self._latest_obs_timing = None
                self._latest_obs_provenance = None
                self._last_inferred_observation_sequence = None

        if hasattr(self, "_action_queue"):
            with self._action_queue_lock:
                self._action_queue = self._new_vla_action_queue()

    def _apply_watchdog_result_locked(self, result: WatchdogResult) -> None:
        """Apply queue invalidation for a watchdog transition."""
        if result.fault_transition:
            self._invalidate_action_state_locked()
            self.get_logger().error(
                f"[WATCHDOG] LATCHED epoch={result.epoch}: {result.reason}. "
                "Action publication is suppressed; explicit restart/rearm required."
            )
        elif result.armed_transition:
            self.get_logger().info(
                f"[WATCHDOG] ARMED epoch={result.epoch}: complete fresh inputs received"
            )

    def _evaluate_watchdog(self) -> bool:
        """Evaluate current input freshness and atomically handle a trip."""
        try:
            snapshot = self.strategy.get_input_snapshot(self.camera_names)
        except Exception as exc:
            self._latch_watchdog(
                f"input snapshot read failed: {type(exc).__name__}: {exc}",
                snapshot=None,
                read_snapshot=False,
            )
            return False
        with self._safety_lock:
            result = self._watchdog.evaluate(snapshot, time.monotonic())
            self._apply_watchdog_result_locked(result)
            return result.publish_allowed

    def _latch_watchdog(
        self,
        reason: str,
        snapshot: InputSnapshot | None = None,
        *,
        read_snapshot: bool = True,
    ) -> None:
        """Latch a non-freshness runtime fault through the same safety path."""
        if snapshot is None and read_snapshot:
            try:
                snapshot = self.strategy.get_input_snapshot(self.camera_names)
            except Exception as exc:
                reason = (
                    f"{reason}; input snapshot read also failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        with self._safety_lock:
            result = self._watchdog.trip(reason, snapshot)
            self._apply_watchdog_result_locked(result)

    def _handle_watchdog_rearm(self, _request, response):
        """Explicitly rearm only after every input recovered and advanced."""
        try:
            snapshot = self.strategy.get_input_snapshot(self.camera_names)
        except Exception as exc:
            message = (
                "rearm input snapshot read failed: "
                f"{type(exc).__name__}: {exc}"
            )
            self._latch_watchdog(
                message,
                snapshot=None,
                read_snapshot=False,
            )
            response.success = False
            response.message = message
            self.get_logger().warn(f"[WATCHDOG] Rearm rejected: {message}")
            return response
        with self._safety_lock:
            success, message = self._watchdog.rearm(snapshot, time.monotonic())
            if not success:
                response.success = False
                response.message = message
                self.get_logger().warn(f"[WATCHDOG] Rearm rejected: {message}")
                return response
            self._invalidate_action_state_locked()
            epoch = self._watchdog.epoch
            try:
                # Keep the safety lock through reset so no timer or inference
                # worker can observe the rearmed epoch before reset completes.
                with self._model_lock:
                    if hasattr(self.model, "reset"):
                        self.model.reset()
            except Exception as exc:
                result = self._watchdog.trip(
                    f"policy reset failed during rearm: {exc}", snapshot
                )
                self._apply_watchdog_result_locked(result)
                response.success = False
                response.message = f"policy reset failed: {exc}"
                return response

        response.success = True
        response.message = f"watchdog rearmed at epoch {epoch}; waiting for a new observation"
        self.get_logger().info(f"[WATCHDOG] {response.message}")
        return response

    def _setup_vla_inference(self) -> None:
        """Initialise ActionQueue and LatencyTracker for VLA / RTC mode."""
        from lerobot_control.latency_stats import LatencyStats

        self._action_queue = self._new_vla_action_queue()
        self._latency_tracker = LatencyStats(maxlen=100)
        self._latest_obs = None
        self._latest_obs_timing = None
        self._latest_obs_provenance = None
        self._last_inferred_observation_sequence = None
        self._obs_lock = threading.Lock()
        self._inference_stop = threading.Event()
        self._rtc_timing_next_sample_id = 0
        self._rtc_pending_cuda_timings: deque[PendingRTCCudaTiming] = deque()
        self._rtc_threshold = int(
            self.rtc_config_yaml.get("queue_trigger_threshold", 30)
        )
        self._rtc_delay_fallback = int(
            self.rtc_config_yaml.get("inference_delay", 4)
        )
        self._rtc_readiness_guided_forwards = int(
            self.rtc_config_yaml.get("readiness_guided_forwards", 5)
        )
        self._rtc_readiness_guard_steps = int(
            self.rtc_config_yaml.get("readiness_latency_guard_steps", 2)
        )
        self._rtc_index_phase_tolerance_steps = int(
            self.rtc_config_yaml.get(
                "readiness_index_phase_tolerance_steps", 1
            )
        )
        self._rtc_scheduler_guard_steps = int(
            self.rtc_config_yaml.get("readiness_scheduler_guard_steps", 1)
        )
        self._rtc_min_guided_overlap_steps = int(
            self.rtc_config_yaml.get("readiness_min_guided_overlap_steps", 3)
        )
        if self._rtc_threshold < 0:
            raise ValueError("rtc.queue_trigger_threshold must be >= 0")
        if self._rtc_delay_fallback < 0:
            raise ValueError("rtc.inference_delay must be >= 0")
        if self._rtc_readiness_guided_forwards != 5:
            raise ValueError("rtc.readiness_guided_forwards must be exactly 5")
        if self._rtc_readiness_guard_steps < 0:
            raise ValueError("rtc.readiness_latency_guard_steps must be >= 0")
        if self._rtc_index_phase_tolerance_steps < 0:
            raise ValueError(
                "rtc.readiness_index_phase_tolerance_steps must be >= 0"
            )
        if self._rtc_scheduler_guard_steps < 0:
            raise ValueError("rtc.readiness_scheduler_guard_steps must be >= 0")
        if self._rtc_min_guided_overlap_steps < 0:
            raise ValueError(
                "rtc.readiness_min_guided_overlap_steps must be >= 0"
            )
        self._vla_warmup_pending = True
        self._vla_policy_ready = False
        self._vla_stale_result_count = 0
        self._vla_seeded = False
        self._rtc_guided_streak = 0
        self._rtc_guided_latencies: deque[float] = deque(
            maxlen=self._rtc_readiness_guided_forwards
        )

    @staticmethod
    def _rtc_chunk_length(original, processed) -> int:
        """Return a validated common action-chunk length."""
        try:
            original_steps = len(original)
            processed_steps = len(processed)
        except (TypeError, AttributeError) as exc:
            raise ValueError("RTC result is not a time-indexed action chunk") from exc
        if original_steps <= 0 or processed_steps <= 0:
            raise ValueError("RTC result contains an empty action chunk")
        if original_steps != processed_steps:
            raise ValueError(
                "RTC original/processed chunk lengths differ: "
                f"{original_steps} != {processed_steps}"
            )
        return int(original_steps)

    @staticmethod
    def _rtc_has_guidance(prev_actions) -> bool:
        """Return whether the forward received at least one leftover action."""
        if prev_actions is None:
            return False
        try:
            return len(prev_actions) > 0
        except TypeError:
            return False

    @staticmethod
    def _rtc_inference_delay_steps(
        *,
        guided_latencies_sec: tuple[float, ...],
        tracked_max_latency_sec: float | None,
        control_freq: float,
        fallback_steps: int,
    ) -> int:
        """Choose RTC's delay estimate from the active readiness window."""
        max_latency = (
            max(guided_latencies_sec)
            if guided_latencies_sec
            else tracked_max_latency_sec
        )
        return (
            math.ceil(max_latency * control_freq)
            if max_latency
            else fallback_steps
        )

    @staticmethod
    def _resolve_rtc_merge_alignment(
        *,
        queue_identity_matches: bool,
        queue_size_before_inference: int,
        queue_size_at_merge: int,
        action_index_before_inference: int,
        action_index_at_merge: int,
        requested_at_monotonic: float,
        merge_at_monotonic: float,
        control_freq: float,
        policy_ready: bool,
        index_phase_tolerance_steps: int,
    ) -> RTCMergeAlignment:
        """Resolve RTC alignment from the exact leftover snapshot and queue.

        ``runtime_sec`` starts when the leftover and queue index are captured
        and ends immediately before merge. Once publication is open, real queue
        consumption is the authoritative merge delay. The wall clock provides
        an upper bound: the timer may be delayed by executor scheduling, but it
        must not consume more actions than the elapsed time can explain.
        Before readiness, publication is closed by design, so consumption must
        be zero and the conservative wall delay is used virtually.
        """
        if (
            not math.isfinite(requested_at_monotonic)
            or not math.isfinite(merge_at_monotonic)
            or merge_at_monotonic < requested_at_monotonic
            or not math.isfinite(control_freq)
            or control_freq <= 0
        ):
            raise ValueError("RTC merge alignment has invalid timing")
        if index_phase_tolerance_steps < 0:
            raise ValueError("RTC index phase tolerance must be >= 0")
        if not queue_identity_matches:
            raise ValueError("RTC action queue changed while inference was running")
        if (
            isinstance(queue_size_before_inference, bool)
            or isinstance(queue_size_at_merge, bool)
            or isinstance(action_index_before_inference, bool)
            or isinstance(action_index_at_merge, bool)
            or not isinstance(queue_size_before_inference, int)
            or not isinstance(queue_size_at_merge, int)
            or not isinstance(action_index_before_inference, int)
            or not isinstance(action_index_at_merge, int)
            or queue_size_before_inference < 0
            or queue_size_at_merge < 0
            or action_index_before_inference < 0
            or action_index_at_merge < action_index_before_inference
        ):
            raise ValueError(
                "RTC queue depth/index regressed or is invalid: "
                f"q_before={queue_size_before_inference}, "
                f"q_merge={queue_size_at_merge}, "
                f"i_before={action_index_before_inference}, "
                f"i_merge={action_index_at_merge}"
            )

        runtime_sec = merge_at_monotonic - requested_at_monotonic
        wall_delay_steps = math.ceil(runtime_sec * control_freq)
        consumed_steps = action_index_at_merge - action_index_before_inference
        queue_depth_delta = queue_size_before_inference - queue_size_at_merge
        if queue_depth_delta != consumed_steps:
            raise ValueError(
                "RTC queue depth/index consumption mismatch: "
                f"depth_delta={queue_depth_delta}, index_delta={consumed_steps}"
            )

        if not policy_ready:
            if consumed_steps != 0:
                raise ValueError(
                    "RTC pre-ready queue was consumed while publication was closed: "
                    f"consumed={consumed_steps} steps"
                )
            merge_delay_steps = wall_delay_steps
        else:
            if queue_size_at_merge < 1:
                raise ValueError("RTC action queue emptied before refill merge")
            phase_steps = runtime_sec * control_freq
            maximum_consumed = (
                math.ceil(phase_steps) + index_phase_tolerance_steps
            )
            if consumed_steps > maximum_consumed:
                raise ValueError(
                    "RTC queue consumption exceeds wall-clock upper bound: "
                    f"consumed={consumed_steps}, maximum={maximum_consumed}, "
                    f"runtime={runtime_sec:.6f}s"
                )
            merge_delay_steps = consumed_steps

        return RTCMergeAlignment(
            requested_at_monotonic=requested_at_monotonic,
            merge_at_monotonic=merge_at_monotonic,
            runtime_sec=runtime_sec,
            wall_delay_steps=wall_delay_steps,
            consumed_steps=consumed_steps,
            merge_delay_steps=merge_delay_steps,
            queue_size_at_request=queue_size_before_inference,
            queue_size_at_merge=queue_size_at_merge,
        )

    @staticmethod
    def _rtc_queue_state_locked(queue) -> tuple[int, int, int]:
        """Read a coherent queue depth, index, and leftover length.

        The caller must hold the node's outer action-queue lock. LeRobot's own
        queue methods do not consistently take its internal lock for every
        getter, so read its canonical fields under that internal lock when they
        are available.
        """
        internal_lock = getattr(queue, "lock", None)
        if internal_lock is not None and all(
            hasattr(queue, name)
            for name in ("queue", "original_queue", "last_index")
        ):
            with internal_lock:
                action_index = queue.last_index
                queue_size = (
                    0
                    if queue.queue is None
                    else len(queue.queue) - action_index
                )
                leftover_size = (
                    0
                    if queue.original_queue is None
                    else len(queue.original_queue) - action_index
                )
            return queue_size, action_index, leftover_size

        queue_size = queue.qsize()
        action_index = queue.get_action_index()
        prev_actions = queue.get_left_over()
        leftover_size = 0 if prev_actions is None else len(prev_actions)
        return queue_size, action_index, leftover_size

    def _capture_rtc_dispatch_locked(
        self,
    ) -> tuple[RTCDispatchSnapshot, object | None]:
        """Capture one internally coherent queue/leftover dispatch snapshot."""
        queue = self._action_queue
        queue_size, action_index, leftover_size = (
            self._rtc_queue_state_locked(queue)
        )
        prev_actions = queue.get_left_over()
        try:
            copied_leftover_size = 0 if prev_actions is None else len(prev_actions)
        except TypeError as exc:
            raise ValueError("RTC leftover is not a time-indexed chunk") from exc
        if (
            isinstance(queue_size, bool)
            or isinstance(action_index, bool)
            or not isinstance(queue_size, int)
            or not isinstance(action_index, int)
            or queue_size < 0
            or action_index < 0
            or leftover_size != queue_size
            or copied_leftover_size != leftover_size
        ):
            raise ValueError(
                "RTC queue/leftover snapshot is incoherent: "
                f"queue={queue_size}, index={action_index}, "
                f"leftover={leftover_size}, copied={copied_leftover_size}"
            )
        dispatch = RTCDispatchSnapshot(
            queue=queue,
            queue_size=queue_size,
            action_index=action_index,
            requested_at_monotonic=time.monotonic(),
        )
        return dispatch, prev_actions

    @staticmethod
    def _rtc_should_wait_for_refill(
        *,
        queue_size: int,
        queue_threshold: int,
        policy_ready: bool,
    ) -> bool:
        """Apply the queue threshold only after publication can drain it."""
        return policy_ready and queue_size > queue_threshold

    @staticmethod
    def _assess_rtc_readiness(
        *,
        chunk_size: int,
        candidate_delay_steps: int,
        guided_latencies_sec: tuple[float, ...],
        control_freq: float,
        queue_threshold: int,
        source_age_sec: float,
        max_action_age_sec: float,
        execution_horizon: int,
        latency_guard_steps: int,
        scheduler_guard_steps: int,
        min_guided_overlap_steps: int,
    ) -> RTCReadinessAssessment:
        """Evaluate whether this chunk can survive the following RTC refill.

        ``q_start`` is the new queue depth after the candidate's validated merge
        delay. It drains to the configured trigger and reserves an additional
        scheduler guard before the next result. That result must arrive with at
        least one action still queued, remain inside the current chunk's exact
        source-age budget, and preserve useful guided overlap.
        """
        if not guided_latencies_sec:
            raise ValueError("RTC readiness requires guided latency samples")
        if any(
            not math.isfinite(latency) or latency < 0
            for latency in guided_latencies_sec
        ):
            raise ValueError("RTC guided latencies must be finite and >= 0")
        if (
            isinstance(candidate_delay_steps, bool)
            or not isinstance(candidate_delay_steps, int)
            or candidate_delay_steps < 0
        ):
            raise ValueError("RTC candidate delay must be an integer >= 0")
        if scheduler_guard_steps < 0:
            raise ValueError("RTC scheduler guard must be >= 0")

        q_start = max(0, chunk_size - candidate_delay_steps)
        wait_steps = max(0, q_start - queue_threshold)
        q_trigger = q_start - wait_steps
        q_required = max(0, q_trigger - scheduler_guard_steps)

        max_guided_latency = max(guided_latencies_sec)
        latency_bound_sec = (
            max_guided_latency + latency_guard_steps / control_freq
        )
        # guard_steps is integral, so this is exactly ceil(f * L_bound) while
        # avoiding a floating-point ceil at an integer boundary.
        refill_delay_bound_steps = (
            math.ceil(max_guided_latency * control_freq) + latency_guard_steps
        )
        coverage_required_steps = refill_delay_bound_steps + 1
        age_at_next_refill_sec = (
            source_age_sec
            + (wait_steps + scheduler_guard_steps) / control_freq
            + latency_bound_sec
        )
        effective_horizon = min(execution_horizon, q_required)
        useful_guided_overlap_steps = max(
            0, effective_horizon - refill_delay_bound_steps
        )

        failures: list[str] = []
        if q_start <= 0:
            failures.append("candidate leaves an empty queue")
        if q_required < coverage_required_steps:
            failures.append(
                "refill coverage "
                f"{q_required} < {coverage_required_steps} steps"
            )
        if age_at_next_refill_sec >= max_action_age_sec:
            failures.append(
                "projected source age "
                f"{age_at_next_refill_sec:.3f}s >= {max_action_age_sec:.3f}s"
            )
        if useful_guided_overlap_steps < min_guided_overlap_steps:
            failures.append(
                "useful guided overlap "
                f"{useful_guided_overlap_steps} < {min_guided_overlap_steps} steps"
            )

        return RTCReadinessAssessment(
            sustainable=not failures,
            candidate_delay_steps=candidate_delay_steps,
            q_start=q_start,
            wait_steps=wait_steps,
            q_trigger=q_trigger,
            scheduler_guard_steps=scheduler_guard_steps,
            q_required=q_required,
            latency_bound_sec=latency_bound_sec,
            refill_delay_bound_steps=refill_delay_bound_steps,
            coverage_required_steps=coverage_required_steps,
            age_at_next_refill_sec=age_at_next_refill_sec,
            useful_guided_overlap_steps=useful_guided_overlap_steps,
            failures=tuple(failures),
        )

    @staticmethod
    def _rtc_assessment_log_fields(
        assessment: RTCReadinessAssessment,
        execution_horizon: int,
    ) -> str:
        """Render the exact readiness margins for operator logs."""
        return (
            f"q_start={assessment.q_start} q_trigger={assessment.q_trigger} "
            f"scheduler_guard={assessment.scheduler_guard_steps} "
            f"q_required={assessment.q_required} "
            f"coverage_required={assessment.coverage_required_steps} "
            f"L_bound={assessment.latency_bound_sec:.3f}s "
            f"projected_age={assessment.age_at_next_refill_sec:.3f}s "
            f"execution_horizon={execution_horizon} "
            f"useful_overlap={assessment.useful_guided_overlap_steps}"
        )

    def _reset_rtc_readiness_streak_locked(self) -> None:
        """Make the current provisional chunk the seed of a new trial."""
        self._rtc_guided_streak = 0
        self._rtc_guided_latencies.clear()

    def _clear_vla_provisional_queue_locked(self) -> None:
        """Drop an unusable pre-ready seed without changing safety epochs."""
        with self._action_queue_lock:
            self._action_queue = self._new_vla_action_queue()
        self._vla_seeded = False
        self._vla_action_source_monotonic = None
        self._vla_action_epoch = None
        if hasattr(self, "_latency_tracker"):
            self._latency_tracker.reset()

    def _trip_rtc_sustainability_locked(self, reason: str) -> bool:
        """Latch and invalidate every action before a post-ready violation."""
        result = self._watchdog.trip(f"RTC sustainability lost: {reason}", None)
        self._apply_watchdog_result_locked(result)
        return False

    def _merge_vla_result_locked(
        self,
        *,
        original,
        processed,
        dispatch: RTCDispatchSnapshot,
        observation_monotonic: float,
        epoch: int,
        stage_timing: RTCMergeStageTiming | None = None,
    ) -> tuple[int, RTCMergeAlignment]:
        """Validate alignment and merge atomically against the captured queue."""
        if stage_timing is not None:
            stage_timing.queue_lock_requested_at_monotonic = time.monotonic()
        with self._action_queue_lock:
            if stage_timing is not None:
                stage_timing.queue_lock_acquired_at_monotonic = time.monotonic()
            merge_at_monotonic = time.monotonic()
            if stage_timing is not None:
                stage_timing.alignment_merge_at_monotonic = merge_at_monotonic
            queue_size_at_merge, action_index_at_merge, leftover_size = (
                self._rtc_queue_state_locked(self._action_queue)
            )
            if leftover_size != queue_size_at_merge:
                raise ValueError(
                    "RTC queue/leftover state diverged before merge: "
                    f"queue={queue_size_at_merge}, leftover={leftover_size}"
                )
            alignment = self._resolve_rtc_merge_alignment(
                queue_identity_matches=self._action_queue is dispatch.queue,
                queue_size_before_inference=dispatch.queue_size,
                queue_size_at_merge=queue_size_at_merge,
                action_index_before_inference=dispatch.action_index,
                action_index_at_merge=action_index_at_merge,
                requested_at_monotonic=dispatch.requested_at_monotonic,
                merge_at_monotonic=merge_at_monotonic,
                control_freq=self.control_freq,
                policy_ready=self._vla_policy_ready,
                index_phase_tolerance_steps=(
                    self._rtc_index_phase_tolerance_steps
                ),
            )
            source_age_at_merge = merge_at_monotonic - observation_monotonic
            if source_age_at_merge > self._watchdog.max_action_age_sec:
                raise ValueError(
                    "RTC result became stale before merge: "
                    f"source_age={source_age_at_merge:.3f}s > "
                    f"{self._watchdog.max_action_age_sec:.3f}s"
                )
            # LeRobot 0.5.1's built-in index check is warning-only and still
            # accepts its caller's delay. We already validated both clocks
            # fail-closed above, so do not rely on that advisory check here.
            self._action_queue.merge(
                original,
                processed,
                alignment.merge_delay_steps,
                None,
            )
            queue_size = self._action_queue.qsize()
            expected_queue_size = max(
                0,
                min(len(original), len(processed))
                - alignment.merge_delay_steps,
            )
            if queue_size != expected_queue_size:
                raise ValueError(
                    "RTC merged queue depth is incoherent: "
                    f"queue={queue_size}, expected={expected_queue_size}"
                )
            if queue_size > 0:
                self._vla_action_source_monotonic = observation_monotonic
                self._vla_action_epoch = epoch
            if stage_timing is not None:
                stage_timing.merge_completed_at_monotonic = time.monotonic()
        return queue_size, alignment

    def _commit_vla_result_locked(
        self,
        *,
        original,
        processed,
        dispatch: RTCDispatchSnapshot,
        observation_monotonic: float,
        epoch: int,
        policy_epoch: int,
        elapsed: float,
        completed_at_monotonic: float,
        guided: bool,
        stage_timing: RTCMergeStageTiming | None = None,
    ) -> bool:
        """Commit an RTC result only after proving sustained real-time margins.

        The first successful forward is an unconditional cold-runtime warm-up
        and is discarded. The next unguided result seeds a provisional queue,
        but publication remains closed. Five consecutive guided refills must
        then satisfy exact queue coverage, source age, and useful-overlap bounds
        before ``POLICY_READY``. Any violation after readiness atomically
        latches the watchdog and clears the queue before another publish.
        """
        if policy_epoch != self._policy_epoch:
            self.get_logger().warn(
                "[RTC] Discarded in-flight result from policy epoch "
                f"{policy_epoch}; current policy epoch is {self._policy_epoch}"
            )
            return False

        if not self._watchdog.is_epoch_current(epoch):
            self.get_logger().warn(
                f"[WATCHDOG] Discarded in-flight result from epoch {epoch}"
            )
            return False

        if (
            not math.isfinite(observation_monotonic)
            or not math.isfinite(completed_at_monotonic)
            or completed_at_monotonic < observation_monotonic
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            result = self._watchdog.trip(
                "RTC result has invalid monotonic timing",
                None,
            )
            self._apply_watchdog_result_locked(result)
            return False

        source_age = completed_at_monotonic - observation_monotonic
        if completed_at_monotonic + 1e-9 < dispatch.requested_at_monotonic:
            result = self._watchdog.trip(
                "RTC result completed before its dispatch snapshot",
                None,
            )
            self._apply_watchdog_result_locked(result)
            return False

        if self._vla_warmup_pending:
            self._vla_warmup_pending = False
            self._vla_policy_ready = False
            self._vla_seeded = False
            self._reset_rtc_readiness_streak_locked()
            self._latency_tracker.reset()
            self.get_logger().info(
                "[RTC] WARMUP_DISCARDED "
                f"epoch={epoch} latency={elapsed * 1000.0:.1f}ms "
                f"source_age={source_age:.3f}s; waiting for a provisional seed"
            )
            return False

        if source_age > self._watchdog.max_action_age_sec:
            self._vla_stale_result_count += 1
            reason = (
                f"source_age={source_age:.3f}s > "
                f"{self._watchdog.max_action_age_sec:.3f}s"
            )
            if self._vla_policy_ready:
                return self._trip_rtc_sustainability_locked(reason)
            self._clear_vla_provisional_queue_locked()
            self._reset_rtc_readiness_streak_locked()
            self.get_logger().warn(
                "[RTC] STALE_RESULT_DISCARDED "
                f"epoch={epoch} {reason} "
                f"(count={self._vla_stale_result_count}); readiness remains closed"
            )
            return False

        try:
            chunk_size = self._rtc_chunk_length(original, processed)
        except ValueError as exc:
            if self._vla_policy_ready:
                return self._trip_rtc_sustainability_locked(str(exc))
            self._clear_vla_provisional_queue_locked()
            self._reset_rtc_readiness_streak_locked()
            self.get_logger().warn(
                f"[RTC] EMPTY_RESULT_DISCARDED epoch={epoch}: {exc}; "
                "readiness remains closed"
            )
            return False

        if not guided:
            if self._vla_policy_ready:
                return self._trip_rtc_sustainability_locked(
                    "a refill ran without leftover-action guidance"
                )
            try:
                queue_size, alignment = self._merge_vla_result_locked(
                    original=original,
                    processed=processed,
                    dispatch=dispatch,
                    observation_monotonic=observation_monotonic,
                    epoch=epoch,
                    stage_timing=stage_timing,
                )
            except Exception as exc:
                self._clear_vla_provisional_queue_locked()
                self._reset_rtc_readiness_streak_locked()
                self.get_logger().warn(
                    "[RTC] EMPTY_RESULT_DISCARDED "
                    f"epoch={epoch}: merge failed: {type(exc).__name__}: {exc}; "
                    "readiness remains closed"
                )
                return False
            expected_queue_size = max(
                0, chunk_size - alignment.merge_delay_steps
            )
            self._reset_rtc_readiness_streak_locked()
            if queue_size <= 0 or queue_size != expected_queue_size:
                self._clear_vla_provisional_queue_locked()
                self.get_logger().warn(
                    "[RTC] EMPTY_RESULT_DISCARDED "
                    f"epoch={epoch} queue={queue_size} expected={expected_queue_size}; "
                    "readiness remains closed"
                )
                return False
            self._vla_seeded = True
            self._latency_tracker.add(alignment.runtime_sec)
            self.metrics.record_inference()
            self.get_logger().info(
                "[RTC] SEED_PROVISIONAL "
                f"epoch={epoch} latency={alignment.runtime_sec * 1000.0:.1f}ms "
                "source_age="
                f"{alignment.merge_at_monotonic - observation_monotonic:.3f}s "
                f"queue={queue_size}; "
                "publication remains closed"
            )
            return False

        if not self._vla_seeded:
            reason = "guided result arrived without a provisional seed"
            if self._vla_policy_ready:
                return self._trip_rtc_sustainability_locked(reason)
            self._clear_vla_provisional_queue_locked()
            self._reset_rtc_readiness_streak_locked()
            self.get_logger().warn(
                f"[RTC] READINESS_REJECTED epoch={epoch}: {reason}"
            )
            return False

        execution_horizon = int(
            self.model.config.rtc_config.execution_horizon
        )
        try:
            queue_size, alignment = self._merge_vla_result_locked(
                original=original,
                processed=processed,
                dispatch=dispatch,
                observation_monotonic=observation_monotonic,
                epoch=epoch,
                stage_timing=stage_timing,
            )
        except Exception as exc:
            reason = f"merge failed: {type(exc).__name__}: {exc}"
            if self._vla_policy_ready:
                return self._trip_rtc_sustainability_locked(reason)
            self._clear_vla_provisional_queue_locked()
            self._reset_rtc_readiness_streak_locked()
            self.get_logger().warn(
                f"[RTC] READINESS_REJECTED epoch={epoch}: {reason}; "
                "publication remains closed"
            )
            return False

        candidate_latencies = tuple(
            [*self._rtc_guided_latencies, alignment.runtime_sec][
                -self._rtc_readiness_guided_forwards :
            ]
        )
        assessment = self._assess_rtc_readiness(
            chunk_size=chunk_size,
            candidate_delay_steps=alignment.merge_delay_steps,
            guided_latencies_sec=candidate_latencies,
            control_freq=self.control_freq,
            queue_threshold=self._rtc_threshold,
            source_age_sec=(
                alignment.merge_at_monotonic - observation_monotonic
            ),
            max_action_age_sec=self._watchdog.max_action_age_sec,
            execution_horizon=execution_horizon,
            latency_guard_steps=self._rtc_readiness_guard_steps,
            scheduler_guard_steps=self._rtc_scheduler_guard_steps,
            min_guided_overlap_steps=self._rtc_min_guided_overlap_steps,
        )
        fields = self._rtc_assessment_log_fields(assessment, execution_horizon)

        if not assessment.sustainable and self._vla_policy_ready:
            return self._trip_rtc_sustainability_locked(
                f"{'; '.join(assessment.failures)}; {fields}"
            )

        self._latency_tracker.add(alignment.runtime_sec)
        self.metrics.record_inference()

        if queue_size != assessment.q_start or queue_size <= 0:
            reason = (
                f"merged queue depth {queue_size} != expected {assessment.q_start}"
            )
            if self._vla_policy_ready:
                return self._trip_rtc_sustainability_locked(reason)
            self._clear_vla_provisional_queue_locked()
            self._reset_rtc_readiness_streak_locked()
            self.get_logger().warn(
                f"[RTC] READINESS_REJECTED epoch={epoch}: {reason}; {fields}"
            )
            return False

        self._vla_seeded = True
        if not assessment.sustainable:
            # Every failed pre-ready proof starts from a fresh unguided seed;
            # rejected guidance must never become the ancestor of readiness.
            self._clear_vla_provisional_queue_locked()
            self._reset_rtc_readiness_streak_locked()
            self.get_logger().warn(
                "[RTC] READINESS_REJECTED "
                f"epoch={epoch}: {'; '.join(assessment.failures)}; {fields}; "
                "streak reset to 0 and publication remains closed"
            )
            return False

        self._rtc_guided_latencies.append(alignment.runtime_sec)
        self._rtc_guided_streak += 1
        if not self._vla_policy_ready:
            if self._rtc_guided_streak < self._rtc_readiness_guided_forwards:
                self.get_logger().info(
                    "[RTC] READINESS_PROGRESS "
                    f"epoch={epoch} guided={self._rtc_guided_streak}/"
                    f"{self._rtc_readiness_guided_forwards} {fields}; "
                    "publication remains closed"
                )
                return False
            self._vla_policy_ready = True
            self.get_logger().info(
                "[RTC] POLICY_READY "
                f"epoch={epoch} guided={self._rtc_guided_streak}/"
                f"{self._rtc_readiness_guided_forwards} "
                f"latency={alignment.runtime_sec * 1000.0:.1f}ms "
                "source_age="
                f"{alignment.merge_at_monotonic - observation_monotonic:.3f}s "
                f"{fields}"
            )
        return True

    def _start_inference_thread(self) -> None:
        """Start the background RTC inference daemon thread."""
        self._watchdog.started_at_monotonic = time.monotonic()
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="rtc-inference",
            daemon=True,
        )
        self._inference_thread.start()

    def _log_rtc_pipeline_timing(
        self,
        *,
        sample_id: int,
        phase: str,
        guided: bool,
        publish_ready_after: bool,
        observation_monotonic: float,
        observation_timing: VLAObservationTiming | None,
        dispatch: RTCDispatchSnapshot,
        model_lock_requested_at: float,
        model_started_at: float,
        predict_started_at: float,
        predict_completed_at: float,
        postprocess_completed_at: float,
        safety_lock_requested_at: float,
        safety_lock_acquired_at: float,
        commit_completed_at: float,
        merge_timing: RTCMergeStageTiming,
    ) -> None:
        """Emit one non-synchronizing timing trace for a debug RTC forward."""
        if not self._rtc_timing_enabled:
            return

        def milliseconds(end: float, start: float) -> float:
            return (end - start) * 1000.0

        def optional_milliseconds(
            end: float | None,
            start: float | None,
        ) -> str:
            if end is None or start is None:
                return "nan"
            return f"{milliseconds(end, start):.3f}"

        if observation_timing is None:
            observation_fields = "obs_read_ms=nan obs_validate_ms=nan obs_preprocess_ms=nan "
            observation_fields += "source_to_ready_ms=nan ready_to_dispatch_ms=nan"
        else:
            observation_fields = (
                "obs_read_ms="
                f"{milliseconds(observation_timing.read_completed_monotonic, observation_timing.read_started_monotonic):.3f} "
                "obs_validate_ms="
                f"{milliseconds(observation_timing.preprocess_started_monotonic, observation_timing.read_completed_monotonic):.3f} "
                "obs_preprocess_ms="
                f"{milliseconds(observation_timing.ready_at_monotonic, observation_timing.preprocess_started_monotonic):.3f} "
                "source_to_ready_ms="
                f"{milliseconds(observation_timing.ready_at_monotonic, observation_monotonic):.3f} "
                "ready_to_dispatch_ms="
                f"{milliseconds(dispatch.requested_at_monotonic, observation_timing.ready_at_monotonic):.3f}"
            )

        self.get_logger().info(
            "[RTC_TIMING] "
            f"sample={sample_id} phase={phase} guided={str(guided).lower()} "
            f"publish_ready_after={str(publish_ready_after).lower()} "
            "merged="
            f"{str(merge_timing.alignment_merge_at_monotonic is not None).lower()} "
            f"{observation_fields} "
            "dispatch_to_model_lock_ms="
            f"{milliseconds(model_lock_requested_at, dispatch.requested_at_monotonic):.3f} "
            "model_lock_wait_ms="
            f"{milliseconds(model_started_at, model_lock_requested_at):.3f} "
            "model_setup_ms="
            f"{milliseconds(predict_started_at, model_started_at):.3f} "
            "predict_ms="
            f"{milliseconds(predict_completed_at, predict_started_at):.3f} "
            "postprocess_ms="
            f"{milliseconds(postprocess_completed_at, predict_completed_at):.3f} "
            "safety_lock_wait_ms="
            f"{milliseconds(safety_lock_acquired_at, safety_lock_requested_at):.3f} "
            "safety_to_queue_lock_ms="
            f"{optional_milliseconds(merge_timing.queue_lock_requested_at_monotonic, safety_lock_acquired_at)} "
            "queue_lock_wait_ms="
            f"{optional_milliseconds(merge_timing.queue_lock_acquired_at_monotonic, merge_timing.queue_lock_requested_at_monotonic)} "
            "queue_merge_ms="
            f"{optional_milliseconds(merge_timing.merge_completed_at_monotonic, merge_timing.queue_lock_acquired_at_monotonic)} "
            "commit_total_ms="
            f"{milliseconds(commit_completed_at, safety_lock_acquired_at):.3f} "
            "request_to_commit_ms="
            f"{milliseconds(commit_completed_at, dispatch.requested_at_monotonic):.3f} "
            "request_to_merge_ms="
            f"{optional_milliseconds(merge_timing.alignment_merge_at_monotonic, dispatch.requested_at_monotonic)} "
            "source_age_at_commit_ms="
            f"{milliseconds(commit_completed_at, observation_monotonic):.3f}"
        )

    def _drain_rtc_cuda_timings(self) -> None:
        """Log completed CUDA event pairs without synchronizing the hot path."""
        if not self._rtc_cuda_timing_enabled:
            return
        while self._rtc_pending_cuda_timings:
            pending = self._rtc_pending_cuda_timings[0]
            try:
                if not pending.end_event.query():
                    break
                self._rtc_pending_cuda_timings.popleft()
                cuda_model_ms = pending.start_event.elapsed_time(
                    pending.end_event
                )
            except Exception as exc:  # noqa: BLE001 - diagnostics are non-fatal.
                self._rtc_pending_cuda_timings.popleft()
                self.get_logger().warn(
                    "[RTC_CUDA_TIMING] dropped sample="
                    f"{pending.sample_id}: {type(exc).__name__}: {exc}"
                )
                continue
            self.get_logger().info(
                "[RTC_CUDA_TIMING] "
                f"sample={pending.sample_id} cuda_model_ms={cuda_model_ms:.3f}"
            )

    def _log_rtc_provenance(
        self,
        *,
        sample_id: int,
        sequence: ObservationSequence,
        provenance: ObservationProvenance | None,
        requested_at_monotonic: float,
        processed_chunk: object,
    ) -> None:
        """Correlate one model chunk with the exact sensor samples it consumed."""
        if not self._rtc_provenance_enabled:
            return
        if provenance is None:
            self.get_logger().error(
                f"[RTC_PROVENANCE] sample={sample_id} missing exact sensor provenance"
            )
            return

        readings = (provenance.joint_state, *provenance.cameras)
        receipt_ages_ms = [
            (requested_at_monotonic - reading.last_seen_monotonic) * 1000.0
            for reading in readings
            if reading.last_seen_monotonic is not None
        ]
        ros_stamps = [
            reading.ros_timestamp
            for reading in readings
            if reading.ros_timestamp is not None and math.isfinite(reading.ros_timestamp)
        ]
        oldest_receipt_age_ms = max(receipt_ages_ms, default=math.nan)
        receipt_skew_ms = (
            max(receipt_ages_ms) - min(receipt_ages_ms)
            if receipt_ages_ms
            else math.nan
        )
        ros_stamp_skew_ms = (
            (max(ros_stamps) - min(ros_stamps)) * 1000.0
            if ros_stamps
            else math.nan
        )

        if isinstance(processed_chunk, torch.Tensor):
            chunk = processed_chunk.detach().to(device="cpu", dtype=torch.float32).numpy()
        else:
            chunk = np.asarray(processed_chunk, dtype=np.float32)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim == 1:
            chunk = chunk[np.newaxis, :]
        chunk = np.ascontiguousarray(chunk)
        first_action = chunk[0] if chunk.size else np.array([], dtype=np.float32)
        first_five_mean = (
            chunk[: min(5, len(chunk))].mean(axis=0)
            if chunk.size
            else np.array([], dtype=np.float32)
        )
        chunk_digest = hashlib.sha256(chunk.tobytes()).hexdigest()[:16]

        def reading_fields(reading) -> str:
            safe_name = reading.name.replace("camera:", "camera_").replace(":", "_")
            stamp = (
                f"{reading.ros_timestamp:.9f}"
                if reading.ros_timestamp is not None
                else "nan"
            )
            age = (
                (requested_at_monotonic - reading.last_seen_monotonic) * 1000.0
                if reading.last_seen_monotonic is not None
                else math.nan
            )
            return (
                f"{safe_name}_seq={reading.sequence} "
                f"{safe_name}_stamp={stamp} "
                f"{safe_name}_receipt_age_ms={age:.3f}"
            )

        first_text = ",".join(f"{value:.6f}" for value in first_action)
        mean_text = ",".join(f"{value:.6f}" for value in first_five_mean)
        sensor_text = " ".join(reading_fields(reading) for reading in readings)
        self.get_logger().info(
            "[RTC_PROVENANCE] "
            f"sample={sample_id} observation_joint_seq={sequence.joint_state} "
            f"oldest_receipt_age_ms={oldest_receipt_age_ms:.3f} "
            f"receipt_skew_ms={receipt_skew_ms:.3f} "
            f"ros_stamp_skew_ms={ros_stamp_skew_ms:.3f} "
            f"chunk_shape={'x'.join(str(size) for size in chunk.shape)} "
            f"chunk_sha256={chunk_digest} "
            f"first_action=[{first_text}] first5_mean=[{mean_text}] "
            f"{sensor_text}"
        )

    def _start_rtc_cuda_timing(self) -> tuple[object, object] | None:
        """Create and record a CUDA event pair without affecting inference."""
        if not self._rtc_cuda_timing_enabled:
            return None
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        except Exception as exc:  # noqa: BLE001 - diagnostics are non-fatal.
            self._rtc_cuda_timing_enabled = False
            self.get_logger().warn(
                "[RTC_CUDA_TIMING] disabled after event setup failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return None
        return start_event, end_event

    def _finish_rtc_cuda_timing(
        self,
        sample_id: int,
        event_pair: tuple[object, object] | None,
    ) -> None:
        """Record the end event and enqueue it for a later non-blocking query."""
        if event_pair is None:
            return
        start_event, end_event = event_pair
        try:
            end_event.record()
        except Exception as exc:  # noqa: BLE001 - diagnostics are non-fatal.
            self._rtc_cuda_timing_enabled = False
            self.get_logger().warn(
                "[RTC_CUDA_TIMING] disabled after end event failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return
        self._rtc_pending_cuda_timings.append(
            PendingRTCCudaTiming(
                sample_id=sample_id,
                start_event=start_event,
                end_event=end_event,
            )
        )

    def _inference_loop(self) -> None:
        """Background inference thread for VLA / RTC mode.

        Continuously predicts the next action chunk whenever ActionQueue depth
        falls to or below the trigger threshold. Postprocessing happens here
        (before merge) so that control_loop can publish directly from the queue
        without any further processing.
        """
        while not self._inference_stop.is_set():
            self._drain_rtc_cuda_timings()
            if not self._evaluate_watchdog():
                time.sleep(0.005)
                continue

            # Capture observation, queue state, and safety epoch atomically with
            # respect to a watchdog trip. Each observation sequence may launch at
            # most one inference.
            with self._safety_lock:
                if not self._watchdog.publish_allowed:
                    continue
                dispatch_error = None
                with self._action_queue_lock:
                    if self._rtc_should_wait_for_refill(
                        queue_size=self._action_queue.qsize(),
                        queue_threshold=self._rtc_threshold,
                        policy_ready=self._vla_policy_ready,
                    ):
                        should_wait = True
                    else:
                        should_wait = False
                        try:
                            dispatch, prev_actions = (
                                self._capture_rtc_dispatch_locked()
                            )
                        except Exception as exc:
                            dispatch_error = exc
                            should_wait = True
                if dispatch_error is not None:
                    result = self._watchdog.trip(
                        "RTC dispatch snapshot failed: "
                        f"{type(dispatch_error).__name__}: {dispatch_error}",
                        None,
                    )
                    self._apply_watchdog_result_locked(result)
                if should_wait:
                    obs_record = None
                else:
                    with self._obs_lock:
                        obs_record = self._latest_obs
                        observation_timing = getattr(
                            self, "_latest_obs_timing", None
                        )
                        observation_provenance = getattr(
                            self, "_latest_obs_provenance", None
                        )
                    if obs_record is not None:
                        obs, sequence, epoch, observation_monotonic = obs_record
                        if (
                            epoch != self._watchdog.epoch
                            or sequence == self._last_inferred_observation_sequence
                        ):
                            obs_record = None
                        else:
                            self._last_inferred_observation_sequence = sequence
                            policy_epoch = self._policy_epoch
                            guided_latencies_snapshot = tuple(
                                self._rtc_guided_latencies
                            )
                            tracked_max_latency = self._latency_tracker.max()

            if obs_record is None:
                time.sleep(0.005)
                continue

            guided = self._rtc_has_guidance(prev_actions)
            timing_enabled = self._rtc_timing_enabled
            provenance_enabled = self._rtc_provenance_enabled
            sample_id = self._rtc_timing_next_sample_id
            if timing_enabled or provenance_enabled:
                self._rtc_timing_next_sample_id += 1

            # Compute inference delay from latency history
            # Once guided stabilization starts, use the same bounded rolling
            # window as the readiness proof. This prevents a slow provisional
            # seed (or LatencyTracker's historical maximum) from permanently
            # overstating RTC delay after five newer stable guided forwards.
            inference_delay = self._rtc_inference_delay_steps(
                guided_latencies_sec=guided_latencies_snapshot,
                tracked_max_latency_sec=tracked_max_latency,
                control_freq=self.control_freq,
                fallback_steps=self._rtc_delay_fallback,
            )

            # Run inference — do NOT use torch.inference_mode():
            # RTCProcessor calls torch.enable_grad() internally for guidance gradients.
            # inference_mode() cannot be overridden and would silently zero all gradients.
            model_lock_requested_at = time.monotonic()
            try:
                with self._model_lock:
                    model_started_at = (
                        time.monotonic()
                        if timing_enabled
                        else model_lock_requested_at
                    )
                    if (
                        not self._watchdog.is_epoch_current(epoch)
                        or policy_epoch != self._policy_epoch
                    ):
                        continue
                    cuda_event_pair = self._start_rtc_cuda_timing()
                    predict_started_at = (
                        time.monotonic()
                        if timing_enabled
                        else model_started_at
                    )
                    raw = self.model.predict_action_chunk(
                        obs,
                        inference_delay=inference_delay,
                        prev_chunk_left_over=prev_actions,
                        execution_horizon=self.model.config.rtc_config.execution_horizon,
                    )
                    predict_completed_at = (
                        time.monotonic() if timing_enabled else 0.0
                    )
                    self._finish_rtc_cuda_timing(sample_id, cuda_event_pair)

                    # Postprocess in the inference thread. Keep it under the model
                    # lock so an explicit rearm cannot reset processors mid-call.
                    original = raw.squeeze(0).clone()
                    if self.postprocessor:
                        processed = self.postprocessor.process_action(raw.squeeze(0))
                    else:
                        processed = original
                    postprocess_completed_at = time.monotonic()
            except Exception as exc:
                import traceback

                self.get_logger().error(f"[RTC] predict_action_chunk failed: {exc}")
                self.get_logger().error(traceback.format_exc())
                self._latch_watchdog(f"RTC inference failed: {exc}")
                continue

            elapsed = postprocess_completed_at - model_lock_requested_at
            # A stale-input trip may occur while the GPU is busy. The epoch check
            # discards that in-flight result before it can enter the RTC queue.
            safety_lock_requested_at = (
                time.monotonic()
                if timing_enabled
                else postprocess_completed_at
            )
            merge_timing = RTCMergeStageTiming() if timing_enabled else None
            with self._safety_lock:
                safety_lock_acquired_at = (
                    time.monotonic()
                    if timing_enabled
                    else safety_lock_requested_at
                )
                if self._vla_warmup_pending:
                    phase = "warmup"
                elif not guided:
                    phase = "seed"
                elif self._vla_policy_ready:
                    phase = "steady"
                else:
                    phase = "readiness"
                publish_ready_after = self._commit_vla_result_locked(
                    original=original,
                    processed=processed,
                    dispatch=dispatch,
                    observation_monotonic=observation_monotonic,
                    epoch=epoch,
                    policy_epoch=policy_epoch,
                    elapsed=elapsed,
                    completed_at_monotonic=time.monotonic(),
                    guided=guided,
                    stage_timing=merge_timing,
                )
                commit_completed_at = (
                    time.monotonic()
                    if timing_enabled
                    else safety_lock_acquired_at
                )
            if timing_enabled:
                self._log_rtc_pipeline_timing(
                    sample_id=sample_id,
                    phase=phase,
                    guided=guided,
                    publish_ready_after=publish_ready_after,
                    observation_monotonic=observation_monotonic,
                    observation_timing=observation_timing,
                    dispatch=dispatch,
                    model_lock_requested_at=model_lock_requested_at,
                    model_started_at=model_started_at,
                    predict_started_at=predict_started_at,
                    predict_completed_at=predict_completed_at,
                    postprocess_completed_at=postprocess_completed_at,
                    safety_lock_requested_at=safety_lock_requested_at,
                    safety_lock_acquired_at=safety_lock_acquired_at,
                    commit_completed_at=commit_completed_at,
                    merge_timing=merge_timing,
                )
            if provenance_enabled:
                self._log_rtc_provenance(
                    sample_id=sample_id,
                    sequence=sequence,
                    provenance=observation_provenance,
                    requested_at_monotonic=dispatch.requested_at_monotonic,
                    processed_chunk=processed,
                )
            self._drain_rtc_cuda_timings()

    def _preprocess_vla_observation(self, observation: dict) -> dict:
        """Preprocess a raw observation for VLA models.

        Follows the official lerobot test convention: build a flat batch dict with
        all observation.* keys plus "task" (as a list of strings), then call the
        preprocessor directly as a callable. The pipeline's to_transition / to_output
        converters handle observation splitting, task → complementary_data routing,
        tokenization, normalization, and device placement in one pass.

        Reference: tests/policies/pi0_pi05/test_pi05_rtc.py
        """
        if self.preprocessor:
            # Build flat batch dict: observation keys + task key (list of strings)
            batch = dict(observation)
            if self.task_description:
                batch["task"] = [self.task_description]
            # preprocessor(batch) → batch_to_transition → _forward → transition_to_batch
            # Output is a flat dict with observation.language.tokens etc. at top level
            observation = self.preprocessor(batch)
        return self._move_to_device(observation)

    def _obs_update(self) -> None:
        """Observation update timer (unified for all models).

        VLA: preprocess and update shared snapshot for background inference thread.
        ACT/Diffusion: preprocess, run select_action, push result to deque.
        """
        timing_enabled = getattr(self, "_rtc_timing_enabled", False)
        callback_started_monotonic = (
            time.monotonic() if timing_enabled else 0.0
        )
        if self._shutting_down:
            return
        if not self._evaluate_watchdog():
            return
        read_started_monotonic = time.monotonic() if timing_enabled else 0.0
        try:
            observation = self.strategy.get_observation(self.camera_names)
        except Exception as exc:
            self._latch_watchdog(
                f"observation read failed: {type(exc).__name__}: {exc}",
                snapshot=None,
                read_snapshot=False,
            )
            return
        read_completed_monotonic = time.monotonic() if timing_enabled else 0.0
        if observation is None:
            return

        sequence = self.strategy.get_last_observation_sequence()
        observation_monotonic = self.strategy.get_last_observation_monotonic()
        observation_provenance = (
            self.strategy.get_last_observation_provenance()
            if self._rtc_provenance_enabled
            else None
        )
        try:
            snapshot = self.strategy.get_input_snapshot(self.camera_names)
        except Exception as exc:
            self._latch_watchdog(
                f"post-observation snapshot read failed: {type(exc).__name__}: {exc}",
                snapshot=None,
                read_snapshot=False,
            )
            return
        if sequence is None or observation_monotonic is None:
            self._latch_watchdog("strategy returned an observation without a sequence", snapshot)
            return
        if self._rtc_provenance_enabled and observation_provenance is None:
            self._latch_watchdog(
                "strategy returned an observation without exact sensor provenance",
                snapshot,
            )
            return

        with self._safety_lock:
            watchdog_result = self._watchdog.accept_observation(sequence, snapshot)
            self._apply_watchdog_result_locked(watchdog_result)
            if not watchdog_result.publish_allowed:
                return
            inference_epoch = watchdog_result.epoch
            policy_epoch = self._policy_epoch

        try:
            if self._is_vla:
                preprocess_started_monotonic = (
                    time.monotonic() if timing_enabled else 0.0
                )
                obs = self._preprocess_vla_observation(observation)
                if timing_enabled:
                    observation_timing = VLAObservationTiming(
                        callback_started_monotonic=callback_started_monotonic,
                        read_started_monotonic=read_started_monotonic,
                        read_completed_monotonic=read_completed_monotonic,
                        preprocess_started_monotonic=preprocess_started_monotonic,
                        ready_at_monotonic=time.monotonic(),
                    )
                else:
                    observation_timing = None
                with self._safety_lock:
                    if (
                        not self._watchdog.is_epoch_current(inference_epoch)
                        or policy_epoch != self._policy_epoch
                    ):
                        return
                    with self._obs_lock:
                        self._latest_obs = (
                            obs,
                            sequence,
                            inference_epoch,
                            observation_monotonic,
                        )
                        self._latest_obs_timing = observation_timing
                        self._latest_obs_provenance = observation_provenance
            else:
                # Keep a reference to the raw (unnormalised) observation so we can
                # capture the joint-state baseline when a new chunk is generated.
                _raw_obs = observation

                # True when the model will actually run a forward pass this tick.
                # ACT uses self._action_queue; Diffusion uses self._queues["action"].
                if hasattr(self.model, "_action_queue"):
                    _will_run_forward = len(self.model._action_queue) == 0
                elif hasattr(self.model, "_queues") and self.model._queues is not None:
                    action_q = self.model._queues.get("action")
                    _will_run_forward = action_q is None or len(action_q) == 0
                else:
                    _will_run_forward = True

                # Detect whether a new action chunk is about to be generated.
                # When the queue is empty, select_action will run the model and fill
                # it with n_action_steps new predictions, all computed relative to
                # the current state.  We capture that state as the delta reference.
                _is_new_chunk = self.use_delta_actions and _will_run_forward

                if self.preprocessor:
                    observation = self.preprocessor(dict(observation))
                observation = self._move_to_device(observation)

                # [DEBUG] Point 1: obs.state after preprocessor (check normalization)
                if self._debug and _is_new_chunk and "observation.state" in observation:
                    _dbg_s = observation["observation.state"]
                    if isinstance(_dbg_s, torch.Tensor):
                        _dbg_s = _dbg_s.cpu().numpy()
                    _dbg_s = np.asarray(_dbg_s).flatten()
                    self.get_logger().info(
                        f"[DEBUG] obs.state (post-preproc): [{', '.join(f'{v:.4f}' for v in _dbg_s)}]"
                    )

                with self._model_lock:
                    if (
                        not self._watchdog.is_epoch_current(inference_epoch)
                        or policy_epoch != self._policy_epoch
                    ):
                        return
                    with torch.inference_mode():
                        if _will_run_forward:
                            _t0 = time.monotonic()
                        action = self.model.select_action(observation)
                        if _will_run_forward:
                            self._latency_tracker.add(time.monotonic() - _t0)
                        # Collect remaining normalized queue items BEFORE postprocessing so
                        # the whole chunk can be denormalized together for delta restore.
                        if (
                            _is_new_chunk
                            and self.use_delta_actions
                            and hasattr(self.model, "_queues")
                        ):
                            _rest_norm = [
                                a.detach().clone()
                                for a in self.model._queues.get("action", [])
                            ]
                        else:
                            _rest_norm = None

                # Capture reference state right after chunk generation
                if _is_new_chunk and "observation.state" in _raw_obs:
                    _s = _raw_obs["observation.state"]
                    if hasattr(_s, "numpy"):
                        _s = (_s.squeeze(0).numpy() if _s.dim() > 1 else _s.numpy())
                    elif hasattr(_s, "cpu"):
                        _s = _s.cpu().numpy()
                    self._delta_ref_state = np.asarray(_s, dtype=np.float64).flatten()

                if self.postprocessor:
                    action = self.postprocessor.process_action(action)

                if isinstance(action, torch.Tensor):
                    if action.dim() > 1:
                        action = action.squeeze(0)
                    action = action.cpu().numpy()

                # [DEBUG] Point 3: action after postprocessor, before delta restore
                if self._debug and _is_new_chunk:
                    self.get_logger().info(
                        f"[DEBUG] action (post-postproc): [{', '.join(f'{v:.4f}' for v in action)}]"
                    )

                # Chunk-level delta restore via shadow queue.
                # The model's internal queue stores normalized tensors; we denormalize
                # the full chunk together, restore delta → absolute, then serve absolute
                # values from a shadow queue so we never re-enter normalized space.
                if self.use_delta_actions:
                    if _is_new_chunk and self._delta_ref_state is not None:
                        if _rest_norm is not None:
                            _rest_denorm = [self._denorm_queue_action(a) for a in _rest_norm]
                            _chunk = np.stack([action] + _rest_denorm) if _rest_denorm else action[np.newaxis]
                        else:
                            _chunk = action[np.newaxis]
                        _abs = restore_delta_chunk(_chunk, self._delta_ref_state, self.action_type, self._delta_exclude_indices)
                        self._abs_shadow_queue = deque(_abs[1:])
                        action = _abs[0]
                    elif self._abs_shadow_queue:
                        action = self._abs_shadow_queue.popleft()
                    elif not hasattr(self.model, "_queues") and self._delta_ref_state is not None:
                        action = restore_delta_chunk(action[np.newaxis], self._delta_ref_state, self.action_type, self._delta_exclude_indices)[0]

                with self._safety_lock:
                    if policy_epoch != self._policy_epoch:
                        self.get_logger().warn(
                            "Discarded in-flight classic-policy result from "
                            f"policy epoch {policy_epoch}; current policy epoch "
                            f"is {self._policy_epoch}"
                        )
                        return
                    if not self._watchdog.is_epoch_current(inference_epoch):
                        self.get_logger().warn(
                            "Discarded in-flight classic-policy result from "
                            f"watchdog epoch {inference_epoch}"
                        )
                        self._invalidate_action_state_locked()
                        return
                    if _will_run_forward:
                        self._classic_chunk_source_monotonic = observation_monotonic
                    action_source = self._classic_chunk_source_monotonic
                    if action_source is None:
                        result = self._watchdog.trip(
                            "classic policy produced an action without a source observation",
                            snapshot,
                        )
                        self._apply_watchdog_result_locked(result)
                        return
                    self._classic_action_deque.append(
                        (action, action_source, inference_epoch)
                    )
                    if _will_run_forward:
                        self.metrics.record_inference()

        except Exception as e:
            import traceback
            self.get_logger().error(f"Observation/inference error: {e}")
            self.get_logger().error(traceback.format_exc())
            self._latch_watchdog(f"observation/inference failed: {e}")

    def _publish_loop(self) -> None:
        """Action publish timer (unified for all models).

        VLA: pop from ActionQueue (filled by background inference thread).
        ACT/Diffusion: pop from deque (filled by _obs_update).
        """
        if self._shutting_down:
            return
        if not self._evaluate_watchdog():
            return
        self.metrics.record_control_loop()

        # Keep the safety lock through the final ROS publish. A concurrent fault
        # transition therefore either happens first (and clears the queue) or
        # waits until this already-authorized publish has completed.
        with self._safety_lock:
            if not self._watchdog.publish_allowed:
                return

            if self._is_vla:
                # Input health (ARMED) and policy readiness are deliberately
                # separate. No warm-up or stale pre-ready result can be popped,
                # authorized, or published.
                if not self._vla_policy_ready:
                    return
                with self._action_queue_lock:
                    action = self._action_queue.get()
                    queue_size = self._action_queue.qsize()
                    action_source = self._vla_action_source_monotonic
                    action_epoch = self._vla_action_epoch
                if self._debug:
                    self._queue_depths.append(queue_size)
                if action is None:
                    self._vla_skip_count += 1
                    result = self._watchdog.trip(
                        "RTC action queue emptied after POLICY_READY",
                        None,
                    )
                    self._apply_watchdog_result_locked(result)
                    return
                if isinstance(action, torch.Tensor):
                    if action.dim() > 1:
                        action = action.squeeze(0)
                    action = action.cpu().numpy()
            else:
                if not self._classic_action_deque:
                    return
                action, action_source, action_epoch = self._classic_action_deque.popleft()

            try:
                snapshot = self.strategy.get_input_snapshot(self.camera_names)
            except Exception as exc:
                result = self._watchdog.trip(
                    f"action authorization snapshot read failed: "
                    f"{type(exc).__name__}: {exc}",
                    None,
                )
                self._apply_watchdog_result_locked(result)
                return

            result = self._watchdog.authorize_action(
                epoch=action_epoch,
                source_monotonic=action_source,
                now=time.monotonic(),
                snapshot=snapshot,
            )
            self._apply_watchdog_result_locked(result)
            if not result.publish_allowed:
                return

            try:
                self._publish_action(action)
            except Exception as e:
                import traceback

                self.get_logger().error(f"Publish error: {e}")
                self.get_logger().error(traceback.format_exc())
                self._latch_watchdog(f"action publication failed: {e}")

    def _reset_delta_state(self) -> None:
        """Reset delta-restore state; call this whenever the model is reloaded."""
        self._delta_ref_state = None
        self._abs_shadow_queue.clear()

    def _denorm_queue_action(self, a: object) -> np.ndarray:
        """Apply postprocessor and convert a queued normalized tensor to a flat numpy array."""
        if self.postprocessor:
            a = self.postprocessor.process_action(a)  # type: ignore[arg-type]
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy().flatten()
        return np.asarray(a).flatten()

    def _move_to_device(self, data):
        """Recursively move tensors to the configured device."""
        if torch.is_tensor(data):
            return data.to(self.device)
        if isinstance(data, dict):
            return {key: self._move_to_device(value) for key, value in data.items()}
        if isinstance(data, tuple):
            return tuple(self._move_to_device(value) for value in data)
        if isinstance(data, list):
            return [self._move_to_device(value) for value in data]
        return data

    def _publish_action(self, action: np.ndarray) -> None:
        """Validate every arm, then publish the complete command set.

        Validation is intentionally transactional: no arm publisher is touched
        until every target has passed shape, finite-value, current-state, delta,
        and absolute-position checks.
        """
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        expected_action_dim = max(
            (arm.get("action_end", 0) for arm in self.arms_config.values()),
            default=0,
        )
        if len(action) != expected_action_dim:
            raise ValueError(
                f"invalid action dimension: got {len(action)}, expected {expected_action_dim}"
            )
        if not np.all(np.isfinite(action)):
            bad = np.flatnonzero(~np.isfinite(action)).tolist()
            raise ValueError(f"action contains non-finite values at indices {bad}")

        covered_indices: list[int] = []
        for arm_name, arm_config in self.arms_config.items():
            start = int(arm_config.get("action_start", 0))
            end = int(arm_config.get("action_end", 0))
            if start < 0 or end <= start or end > expected_action_dim:
                raise ValueError(
                    f"invalid action slice for arm {arm_name}: [{start}:{end}]"
                )
            covered_indices.extend(range(start, end))
        if sorted(covered_indices) != list(range(expected_action_dim)):
            raise ValueError("arm action slices must exactly cover each output index once")

        current_positions = self.strategy.get_current_joint_positions()
        if not current_positions:
            raise ValueError("current joint positions unavailable at publish time")
        joint_order = self.joint_names_config.get(
            "controller_joint_order",
            self.joint_names_config.get("joint_order", []),
        )

        monitor_obs_parts: list[np.ndarray] = []
        monitor_cmd_parts: list[np.ndarray] = []
        pending_messages: list[tuple[str, Float64MultiArray]] = []
        processed_full = np.empty_like(action)

        for arm_name, arm_config in self.arms_config.items():
            start_idx = arm_config.get("action_start", 0)
            end_idx = arm_config.get("action_end", len(action))
            ros_prefix = arm_config.get("ros_prefix", arm_name)

            model_order_action = action[start_idx:end_idx].copy()

            current_names = [
                f"{ros_prefix}_{joint_order[i]}" for i in range(len(model_order_action))
            ]
            missing = [name for name in current_names if name not in current_positions]
            if missing:
                raise ValueError("joint positions missing at publish time: " + ", ".join(missing))
            arm_current = np.array([current_positions[name] for name in current_names])
            if not np.all(np.isfinite(arm_current)):
                raise ValueError(f"current joint positions are non-finite for arm {arm_name}")

            # Delta restore is done upstream in _obs_update (chunk-level), so the
            # raw model target is already absolute. Validate it in controller
            # order *before* delta limiting: otherwise an impossible target can
            # be hidden by a small per-cycle clamp and walk the robot toward the
            # invalid target over repeated cycles.
            raw_controller_action = self.action_limiter.reorder(model_order_action)
            raw_controller_action = self._saturate_mechanical_stops(
                current_names,
                raw_controller_action,
            )
            raw_controller_action = self._validate_absolute_joint_targets(
                current_names,
                raw_controller_action,
                stage="raw absolute",
            )

            arm_action = self.action_limiter.process_controller_order(
                raw_controller_action,
                arm_current,
            )
            if not np.all(np.isfinite(arm_action)):
                raise ValueError(f"processed action is non-finite for arm {arm_name}")
            arm_action = self._validate_absolute_joint_targets(
                current_names,
                arm_action,
                stage="final command",
            )
            processed_full[start_idx:end_idx] = arm_action

            if self._debug:
                formatted = ", ".join(f"{v:.4f}" for v in arm_action)
                self.get_logger().info(f"[DEBUG] cmd [{arm_name}]: [{formatted}]")

            msg = Float64MultiArray()
            msg.data = arm_action.tolist()
            if arm_name not in self.arm_publishers:
                raise ValueError(f"action publisher missing for arm {arm_name}")
            pending_messages.append((arm_name, msg))

            if self._monitor_enable:
                if arm_current is not None:
                    monitor_obs_parts.append(arm_current)
                monitor_cmd_parts.append(arm_action)

        # Only publish after every arm passed every check above.
        for arm_name, msg in pending_messages:
            self.arm_publishers[arm_name].publish(msg)

        if self._monitor_enable and monitor_cmd_parts:
            self._publish_monitor(
                obs_state=np.concatenate(monitor_obs_parts) if monitor_obs_parts else np.zeros_like(action),
                raw_output=action,
                control_cmd=processed_full,
            )

        # Debug: track smoothness
        if self._smooth_tracker is not None:
            self._smooth_tracker.record(processed_full)
        self.metrics.record_action_output()
        self._has_published = True

    def _saturate_mechanical_stops(
        self,
        joint_names: list[str],
        targets: np.ndarray,
    ) -> np.ndarray:
        """Clamp configured targets to hard limits inside a bounded margin."""
        if (
            not self._enforce_joint_position_limits
            or not self._saturate_joint_targets
        ):
            return targets

        targets = np.asarray(targets, dtype=np.float64).reshape(-1).copy()
        for index, joint_name in enumerate(joint_names):
            if joint_name not in self._saturate_joint_targets:
                continue
            lower, upper = self._joint_position_limits[joint_name]
            target = targets[index]
            if lower <= target <= upper:
                continue
            margin = self._saturate_joint_margins[joint_name]
            if target < lower - margin or target > upper + margin:
                # Leave larger violations untouched so the absolute validator
                # latches instead of silently accepting an impossible target.
                continue
            targets[index] = float(np.clip(target, lower, upper))
            previous = self._saturation_counts.get(joint_name, 0)
            self._saturation_counts[joint_name] = previous + 1
            if previous == 0:
                self.get_logger().warn(
                    f"[SATURATE] {joint_name}={target:.9f} clamped to "
                    f"[{lower:.9f}, {upper:.9f}] within margin {margin:.9f}; "
                    "further clamps are counted in the stats block"
                )
        return targets

    def _validate_absolute_joint_targets(
        self,
        joint_names: list[str],
        targets: np.ndarray,
        *,
        stage: str,
    ) -> np.ndarray:
        """Reject out-of-range targets and clip numerical tolerance to the bound."""
        targets = np.asarray(targets, dtype=np.float64).reshape(-1).copy()
        if len(targets) != len(joint_names):
            raise ValueError(
                f"{stage} target dimension mismatch: "
                f"got {len(targets)}, expected {len(joint_names)}"
            )
        if not np.all(np.isfinite(targets)):
            raise ValueError(f"{stage} joint target is non-finite")

        if not self._enforce_joint_position_limits:
            return targets

        for index, (joint_name, target) in enumerate(
            zip(joint_names, targets, strict=True)
        ):
            lower, upper = self._joint_position_limits[joint_name]
            if (
                target < lower - self._joint_limit_tolerance
                or target > upper + self._joint_limit_tolerance
            ):
                raise ValueError(
                    f"{stage} joint target outside absolute limit: "
                    f"{joint_name}={target:.9f}, "
                    f"allowed=[{lower:.9f}, {upper:.9f}]"
                )
            targets[index] = float(np.clip(target, lower, upper))
        return targets

    def _publish_monitor(
        self,
        obs_state: np.ndarray,
        raw_output: np.ndarray,
        control_cmd: np.ndarray,
    ) -> None:
        """Publish monitor topics for real-time inference visualization."""
        obs_msg = Float64MultiArray()
        obs_msg.data = obs_state.tolist()
        self._monitor_obs_pub.publish(obs_msg)

        raw_msg = Float64MultiArray()
        raw_msg.data = raw_output.tolist()
        self._monitor_raw_pub.publish(raw_msg)

        cmd_msg = Float64MultiArray()
        cmd_msg.data = control_cmd.tolist()
        self._monitor_cmd_pub.publish(cmd_msg)

    def _log_input_stats(self) -> None:
        """Periodically log input reception statistics with windowed rates."""
        stats = self.metrics.get_stats()
        if stats["elapsed_sec"] < 1.0:
            return  # Wait for enough data

        # Get frame counters from shared memory workers
        try:
            frame_counters: dict[str, int] = self.strategy.get_frame_counters() or {}
        except Exception as exc:
            self._latch_watchdog(
                f"camera counter snapshot read failed: {type(exc).__name__}: {exc}",
                snapshot=None,
                read_snapshot=False,
            )
            return

        # Compute windowed rates (delta since last log)
        now = time.time()
        if self._prev_log_time is not None:
            dt = max(now - self._prev_log_time, 0.001)
        else:
            dt = stats["elapsed_sec"]

        joint_hz = (stats["joint_count"] - self._prev_joint_count) / dt
        control_hz = (stats["control_loop_count"] - self._prev_control_count) / dt
        inference_delta = stats["inference_count"] - self._prev_inference_count
        inference_hz = inference_delta / dt
        action_output_hz = (stats["action_output_count"] - self._prev_action_output_count) / dt

        camera_hz: dict[str, float] = {}
        camera_delta: dict[str, int] = {}
        for name, count in frame_counters.items():
            prev = self._prev_frame_counters.get(name, 0)
            camera_delta[name] = count - prev
            camera_hz[name] = camera_delta[name] / dt

        # Store snapshot for next window
        self._prev_log_time = now
        self._prev_joint_count = stats["joint_count"]
        self._prev_control_count = stats["control_loop_count"]
        self._prev_inference_count = stats["inference_count"]
        self._prev_action_output_count = stats["action_output_count"]
        self._prev_frame_counters = dict(frame_counters)

        # Find bottleneck camera: compare each camera against its own expected fps,
        # not control_freq (camera target rate is independent of the control loop).
        bottleneck_name = None
        if not self.echo_topic_only and camera_hz:
            slow_cameras = [
                (name, hz)
                for name, hz in camera_hz.items()
                if hz < self._expected_camera_fps.get(name, 30.0) * 2 / 3
            ]
            if slow_cameras:
                bottleneck_name = min(slow_cameras, key=lambda x: x[1])[0]

        # Common header: joint state + cameras
        logger = self.get_logger()
        logger.info(f"-- Stats ({dt:.0f}s) " + "-" * 30)
        logger.info(f"  Joint State  {joint_hz:7.1f} Hz")
        for name in sorted(camera_hz.keys()):
            hz = camera_hz[name]
            delta = camera_delta.get(name, 0)
            marker = "  << bottleneck" if name == bottleneck_name else ""
            logger.info(f"  {name:12s}  {hz:7.1f} Hz  (+{delta} frames){marker}")

        if not self.echo_topic_only:
            if self._is_vla:
                self._log_stats_vla(logger, dt, stats, inference_hz, action_output_hz, bottleneck_name, camera_hz)
            else:
                self._log_stats_classic(logger, dt, stats, control_hz, inference_hz, action_output_hz, bottleneck_name, camera_hz)

    def _log_stats_common(self, logger, inference_hz, action_output_hz, stats) -> None:
        """Log model-agnostic stats shared across all model types."""
        logger.info(f"  Inference FPS{inference_hz:7.1f} Hz  ({stats['inference_count']} total)")
        logger.info(f"  Action FPS   {action_output_hz:7.1f} Hz")
        for joint_name in sorted(self._saturation_counts):
            logger.info(
                f"  Saturated    {self._saturation_counts[joint_name]:7d}    "
                f"{joint_name}"
            )
        if hasattr(self, "_latency_tracker"):
            lat_mean = self._latency_tracker.mean()
            lat_std = self._latency_tracker.std()
            lat_p95 = self._latency_tracker.p95() or 0.0
            if lat_mean > 0:
                logger.info(
                    f"  Infer latency mean={lat_mean * 1000:.1f}ms  "
                    f"std={lat_std * 1000:.1f}ms  p95={lat_p95 * 1000:.1f}ms"
                )

    def _log_stats_vla(self, logger, _dt, stats, inference_hz, action_output_hz, bottleneck_name, camera_hz) -> None:
        """Log VLA (RTC) specific stats."""
        self._log_stats_common(logger, inference_hz, action_output_hz, stats)

        # VLA: additionally log queue size
        if hasattr(self, "_latency_tracker"):
            lat_mean = self._latency_tracker.mean()
            if lat_mean > 0 and hasattr(self, "_action_queue"):
                queue_size = self._action_queue.qsize()
                logger.info(f"  VLA queue    {queue_size}")

            # Debug: Action FPS, Eff ctrl Hz, queue depth stats, smoothness
            if self._debug and lat_mean > 0:
                cs = getattr(self.model.config, "chunk_size", 0)
                eh = getattr(self.model.config.rtc_config, "execution_horizon", 0)
                action_fps = cs / lat_mean
                eff_ctrl_hz = action_fps * eh / cs if cs > 0 else 0
                logger.info(f"  [DEBUG] Action FPS {action_fps:.1f}  Eff ctrl Hz {eff_ctrl_hz:.1f}")

        if self._debug and self._queue_depths:
            depths = np.array(self._queue_depths)
            skip_pct = self._vla_skip_count / max(len(self._queue_depths) + self._vla_skip_count, 1) * 100
            logger.info(f"  [DEBUG] Queue depth min={depths.min()} mean={depths.mean():.0f} max={depths.max()} skip={skip_pct:.1f}%")
            self._queue_depths.clear()
            self._vla_skip_count = 0

        if self._debug and self._smooth_tracker is not None:
            smooth = self._smooth_tracker.get_stats()
            if smooth:
                logger.info(
                    f"  [DEBUG] Action D mean={smooth['delta_mean']:.4f} "
                    f"std={smooth['delta_std']:.4f} max={smooth['delta_max']:.4f} "
                    f"jerk={smooth['jerk_mean']:.4f}"
                )

        if bottleneck_name is not None:
            exp = self._expected_camera_fps.get(bottleneck_name, 30.0)
            logger.warn(
                f"  '{bottleneck_name}' is slow: {camera_hz[bottleneck_name]:.1f} Hz"
                f" (threshold: {exp * 2 / 3:.0f} Hz, expected: {exp:.0f} Hz)"
            )

    def _log_stats_classic(self, logger, _dt, stats, _control_hz, inference_hz, action_output_hz, bottleneck_name, camera_hz) -> None:
        """Log non-VLA (ACT/Diffusion) stats."""
        self._log_stats_common(logger, inference_hz, action_output_hz, stats)

        if self._debug and self._smooth_tracker is not None:
            smooth = self._smooth_tracker.get_stats()
            if smooth:
                logger.info(
                    f"  [DEBUG] Action D mean={smooth['delta_mean']:.4f} "
                    f"std={smooth['delta_std']:.4f} max={smooth['delta_max']:.4f} "
                    f"jerk={smooth['jerk_mean']:.4f}"
                )

        if bottleneck_name is not None:
            exp = self._expected_camera_fps.get(bottleneck_name, 30.0)
            logger.warn(
                f"  '{bottleneck_name}' is slow: {camera_hz[bottleneck_name]:.1f} Hz"
                f" (threshold: {exp * 2 / 3:.0f} Hz, expected: {exp:.0f} Hz)"
            )

    def reset_policy(self) -> None:
        """Reset policy state."""
        if not hasattr(self, "model"):
            return
        self.get_logger().info("Resetting policy state...")
        with self._safety_lock:
            self._invalidate_action_state_locked()
            # Keep safety publication/commit exclusion in force until the model
            # reset completes. An older forward may finish before this acquires
            # the model lock, but its captured policy epoch can no longer commit.
            with self._model_lock:
                if hasattr(self.model, "reset"):
                    self.model.reset()
        self.get_logger().info("Policy state reset complete")

    def get_input_stats(self) -> dict:
        """Get input reception statistics."""
        return self.metrics.get_stats()

    def _publish_hold_position(self) -> None:
        """Publish current joint positions to hold the robot in place on shutdown."""
        if not hasattr(self, "arm_publishers"):
            return
        current = self.strategy.get_current_joint_positions()
        if not current:
            return
        joint_order = self.joint_names_config.get(
            "controller_joint_order",
            self.joint_names_config.get("joint_order", []),
        )
        for arm_name, arm_config in self.arms_config.items():
            ros_prefix = arm_config.get("ros_prefix", arm_name)
            start_idx = arm_config.get("action_start", 0)
            end_idx = arm_config.get("action_end", len(joint_order))
            arm_dim = end_idx - start_idx
            arm_joints = joint_order[:arm_dim]
            names = [f"{ros_prefix}_{joint}" for joint in arm_joints]
            if len(names) != arm_dim or any(name not in current for name in names):
                self.get_logger().error(
                    f"Shutdown: cannot hold {arm_name}; current joint state is incomplete"
                )
                continue
            positions = [current[name] for name in names]
            if not np.all(np.isfinite(positions)):
                self.get_logger().error(
                    f"Shutdown: cannot hold {arm_name}; current joint state is non-finite"
                )
                continue
            msg = Float64MultiArray()
            msg.data = positions
            if arm_name in self.arm_publishers:
                self.arm_publishers[arm_name].publish(msg)
        self.get_logger().info("Shutdown: hold-position command sent to controllers")

    def destroy_node(self) -> None:
        """Cleanup timers, inference thread, strategy, and destroy node."""
        # Block any new publishes first — timers may still fire during executor shutdown
        self._shutting_down = True

        # Cancel timers before stopping the inference thread so no new callbacks
        # are scheduled while we wait for the thread to join.
        for timer_name in ("_obs_timer", "_publish_timer", "_stats_timer"):
            timer = getattr(self, timer_name, None)
            if timer:
                timer.cancel()

        # Stop background RTC inference thread
        if hasattr(self, "_inference_stop"):
            self._inference_stop.set()
        if hasattr(self, "_inference_thread"):
            self._inference_thread.join(timeout=2.0)

        # Hold position before publisher is torn down — only if we actually
        # commanded the robot at least once during this session.
        if (
            not self.echo_topic_only
            and self._has_published
            and self._evaluate_watchdog()
        ):
            self._publish_hold_position()

        self.strategy.cleanup()
        super().destroy_node()


def main(args=None):
    """Main entry point with single-threaded executor."""
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = LeRobotInferenceNode()

        # Use MultiThreadedExecutor: VLA mode needs 3+ threads
        # (obs timer, publish timer, stats timer, joint subscription)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)

        node.get_logger().info("Starting inference loop...")
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if executor:
            executor.shutdown()
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
