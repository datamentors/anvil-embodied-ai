"""
Multi-Process Inference Strategy

Uses separate worker processes for image acquisition, providing true
parallelism (no GIL contention), process isolation, and crash resilience.

Architecture:
- Image Worker Processes: One per camera, subscribe to topics, decompress JPEG,
  write to shared memory
- Main Process: Read from shared memory, run model inference, publish actions
"""

import math
import multiprocessing as mp
import threading
import time
from typing import Any

import torch
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState

from ..image_worker import (
    joint_state_qos_profile,
    run_image_worker,
    run_joint_state_worker,
)
from ..input_watchdog import InputSnapshot, ObservationSequence, SensorReading
from ..shared_image_buffer import SharedImageBuffer, SharedJointStateBuffer

JOINT_STATE_BUFFER_NAME = "lerobot_joint_state"


class MultiProcessStrategy:
    """
    Multi-process strategy using shared memory and worker processes.

    Provides better process isolation - worker crashes don't affect the
    main inference process. This is the default mode (mode: mp).
    """

    def __init__(self):
        self._node = None
        self._config = None
        self._camera_names: list[str] = []
        self._camera_mapping: dict[str, str] = {}
        self._joint_names_config: dict = {}
        self._image_shape: tuple = (480, 640, 3)

        # Shared memory buffer
        self._image_buffer: SharedImageBuffer | None = None
        self._joint_buffer: SharedJointStateBuffer | None = None

        # Worker processes
        self._worker_processes: list[mp.Process] = []
        self._stop_event: mp.Event | None = None

        # Joint state (handled in main process - lightweight)
        self._joint_positions: dict[str, float] | None = None
        self._joint_velocities: dict[str, float] | None = None
        self._joint_efforts: dict[str, float] | None = None
        self._joint_timestamp: float | None = None
        self._joint_received_monotonic: float | None = None
        self._joint_sequence: int = 0
        self._last_consumed_joint_sequence: int = 0
        self._joint_lock = threading.Lock()
        self._required_joint_names: tuple[str, ...] = ()
        self._joint_errors: tuple[str, ...] = ()
        self._last_observation_sequence: ObservationSequence | None = None
        self._last_observation_monotonic: float | None = None
        self._max_sensor_skew_sec: float = 0.10
        self._joint_state_worker_enabled = False
        self._joint_state_topic = ""
        self._joint_worker_last_counter = 0
        self._joint_worker_process: mp.Process | None = None
        self._joint_worker_refresh_lock = threading.Lock()

        # Metrics tracker (set via setup)
        self._metrics = None

        # Status tracking
        self._last_incomplete_reason: str = ""

    def setup(
        self,
        node: Any,
        config: dict,
        camera_mapping: dict[str, str],
        joint_names_config: dict,
        joint_state_topic: str,
        image_shape: tuple,
        metrics: Any = None,
        callback_group: Any = None,
        debug_image_dir: str | None = None,
    ) -> None:
        """Initialize shared memory and start worker processes."""
        self._node = node
        self._config = config
        self._camera_mapping = camera_mapping
        self._camera_names = list(camera_mapping.values())
        self._joint_names_config = joint_names_config
        self._image_shape = image_shape
        self._metrics = metrics
        self._callback_group = callback_group
        self._debug_image_dir = debug_image_dir
        self._joint_state_topic = joint_state_topic

        runtime_config = config.get("runtime", {})
        if not isinstance(runtime_config, dict):
            raise ValueError("runtime must be a mapping")
        joint_state_worker = runtime_config.get("joint_state_worker", False)
        if not isinstance(joint_state_worker, bool):
            raise ValueError("runtime.joint_state_worker must be a boolean")
        self._joint_state_worker_enabled = joint_state_worker
        if self._joint_state_worker_enabled:
            self._validate_joint_state_worker_mode(config)

        watchdog_config = config.get("watchdog", {})
        try:
            max_sensor_skew_sec = float(watchdog_config.get("max_sensor_skew_sec", 0.10))
        except (TypeError, ValueError) as exc:
            raise ValueError("watchdog.max_sensor_skew_sec must be finite and > 0") from exc
        if not math.isfinite(max_sensor_skew_sec) or max_sensor_skew_sec <= 0:
            raise ValueError("watchdog.max_sensor_skew_sec must be finite and > 0")
        self._max_sensor_skew_sec = max_sensor_skew_sec

        obs_prefix = joint_names_config.get("observation_prefix", "follower")
        separator = joint_names_config.get("separator", "_")
        arm_mapping = joint_names_config.get("arm_mapping", {"l": "left", "r": "right"})
        model_joint_order = joint_names_config.get("model_joint_order", [])
        self._required_joint_names = tuple(
            f"{obs_prefix}{separator}{arm_key}{separator}{joint_id}"
            for arm_key in sorted(arm_mapping)
            for joint_id in model_joint_order
        )

        # Create shared memory buffers
        self._setup_shared_memory()

        # Start sensor worker processes
        self._start_workers()

        # The opt-in worker owns the only joint-state subscription when enabled.
        if not self._joint_state_worker_enabled:
            self._setup_joint_subscription(joint_state_topic)

        self._node.get_logger().info(
            "MultiProcessStrategy initialized with "
            f"{len(self._camera_mapping)} image workers and "
            f"joint_state_worker={self._joint_state_worker_enabled}"
        )

    def _setup_shared_memory(self) -> None:
        """Create shared memory buffers for all cameras."""
        self._node.get_logger().info("Setting up shared memory buffers...")

        self._image_buffer = SharedImageBuffer(
            camera_names=self._camera_names,
            image_shape=self._image_shape,
            create=True,
        )

        if self._joint_state_worker_enabled:
            self._joint_buffer = SharedJointStateBuffer(
                create=True,
                buffer_name=JOINT_STATE_BUFFER_NAME,
            )

        self._node.get_logger().info(f"Created shared memory for {len(self._camera_names)} cameras")

    @staticmethod
    def _validate_joint_state_worker_mode(config: dict) -> None:
        """Require an explicit opt-in when worker isolation controls live sinks."""
        runtime = config.get("runtime", {})
        allow_live = runtime.get("allow_live_joint_state_worker", False)
        if not isinstance(allow_live, bool):
            raise ValueError(
                "runtime.allow_live_joint_state_worker must be a boolean"
            )
        arms = config.get("arms")
        if not isinstance(arms, dict) or not arms:
            raise ValueError(
                "runtime.joint_state_worker requires at least one configured arm"
            )

        invalid_topics: list[str] = []
        for arm_name, arm_config in arms.items():
            command_topic = (
                arm_config.get("command_topic") if isinstance(arm_config, dict) else None
            )
            if (
                not isinstance(command_topic, str)
                or not command_topic.startswith("/debug/")
                or len(command_topic) <= len("/debug/")
            ):
                invalid_topics.append(f"arms.{arm_name}.command_topic={command_topic!r}")

        if invalid_topics and not allow_live:
            raise ValueError(
                "runtime.joint_state_worker with live command topics requires "
                "runtime.allow_live_joint_state_worker=true: "
                + ", ".join(invalid_topics)
            )

    def _start_workers(self) -> None:
        """Start image worker processes."""
        self._node.get_logger().info("Starting image worker processes...")

        # Use 'spawn' context for clean subprocess start
        ctx = mp.get_context("spawn")
        self._stop_event = ctx.Event()
        self._worker_processes = []

        for topic, camera_name in self._camera_mapping.items():
            p = ctx.Process(
                target=run_image_worker,
                args=(topic, camera_name, self._image_shape),
                kwargs={
                    "stop_event": self._stop_event,
                    "debug_dir": self._debug_image_dir,
                },
                name=f"image_worker_{camera_name}",
            )
            p.start()
            self._worker_processes.append(p)
            self._node.get_logger().info(f"Started worker: {topic} -> {camera_name} (PID: {p.pid})")

        if self._joint_state_worker_enabled:
            p = ctx.Process(
                target=run_joint_state_worker,
                args=(self._joint_state_topic,),
                kwargs={
                    "buffer_name": JOINT_STATE_BUFFER_NAME,
                    "stop_event": self._stop_event,
                },
                name="joint_state_worker",
            )
            p.start()
            self._worker_processes.append(p)
            self._joint_worker_process = p
            self._node.get_logger().info(
                f"Started worker: {self._joint_state_topic} -> joint states (PID: {p.pid})"
            )

        # Give workers time to connect to shared memory
        time.sleep(0.5)

    def _setup_joint_subscription(self, joint_state_topic: str) -> None:
        """Setup joint state subscription (runs in main process)."""
        self._node.create_subscription(
            JointState,
            joint_state_topic,
            self._joint_callback,
            joint_state_qos_profile(),
            callback_group=self._callback_group,
        )
        self._node.get_logger().info(f"Subscribed to: {joint_state_topic}")

    def _joint_callback(self, msg: JointState) -> None:
        """Process joint state (lightweight, no GIL issue)."""
        # Capture callback ingress, before validation or metric work can delay
        # the timestamp used for freshness and action provenance.
        received_monotonic = time.monotonic()

        # Record metrics
        if self._metrics:
            self._metrics.record_joint_state()

        parsed = self._parse_joint_message(msg)
        with self._joint_lock:
            self._store_joint_sample_locked(
                *parsed,
                received_monotonic=received_monotonic,
                sequence=self._joint_sequence + 1,
            )

    @staticmethod
    def _parse_joint_message(
        msg: JointState,
    ) -> tuple[
        dict[str, float],
        dict[str, float] | None,
        dict[str, float] | None,
        float,
        tuple[str, ...],
    ]:
        """Apply the canonical parser used by both joint-state ingress modes."""
        errors: list[str] = []
        if len(msg.name) != len(msg.position):
            errors.append(f"name/position length mismatch ({len(msg.name)} != {len(msg.position)})")
        if len(set(msg.name)) != len(msg.name):
            errors.append("duplicate joint names")
        if msg.velocity and len(msg.name) != len(msg.velocity):
            errors.append(f"name/velocity length mismatch ({len(msg.name)} != {len(msg.velocity)})")
        if msg.effort and len(msg.name) != len(msg.effort):
            errors.append(f"name/effort length mismatch ({len(msg.name)} != {len(msg.effort)})")

        positions = dict(zip(msg.name, msg.position, strict=False))
        velocities = dict(zip(msg.name, msg.velocity, strict=False)) if msg.velocity else None
        efforts = dict(zip(msg.name, msg.effort, strict=False)) if msg.effort else None
        ros_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        return positions, velocities, efforts, ros_timestamp, tuple(errors)

    def _store_joint_sample_locked(
        self,
        positions: dict[str, float],
        velocities: dict[str, float] | None,
        efforts: dict[str, float] | None,
        ros_timestamp: float,
        errors: tuple[str, ...],
        *,
        received_monotonic: float,
        sequence: int,
    ) -> None:
        """Replace the live joint sample while ``_joint_lock`` is held."""
        self._joint_positions = positions
        self._joint_velocities = velocities
        self._joint_efforts = efforts
        self._joint_timestamp = ros_timestamp
        self._joint_received_monotonic = received_monotonic
        self._joint_sequence = sequence
        self._joint_errors = errors

    def _refresh_joint_state_from_worker(self) -> None:
        """Import the latest coherent worker sample without consuming the slot."""
        if not self._joint_state_worker_enabled:
            return
        if self._joint_buffer is None:
            raise RuntimeError("joint-state worker is enabled without its shared-memory buffer")
        if self._joint_worker_process is None:
            raise RuntimeError("joint-state worker process is not available")
        if self._joint_worker_process.exitcode is not None:
            raise RuntimeError(
                "joint-state worker exited unexpectedly "
                f"(exitcode={self._joint_worker_process.exitcode})"
            )

        # The inference, watchdog, and control callbacks can all request a
        # refresh concurrently. Serialize slot reads and cache by public counter
        # so each CDR payload is deserialized and parsed at most once.
        with self._joint_worker_refresh_lock:
            payload, received_monotonic, counter = self._joint_buffer.read()
            if counter == 0:
                return
            if received_monotonic is None:
                raise RuntimeError("joint-state worker sample has no receive timestamp")
            if counter < self._joint_worker_last_counter:
                raise RuntimeError(
                    "joint-state worker counter regressed "
                    f"({counter} < {self._joint_worker_last_counter})"
                )
            if counter == self._joint_worker_last_counter:
                return

            msg = deserialize_message(payload, JointState)
            parsed = self._parse_joint_message(msg)
            metric_delta = counter - self._joint_worker_last_counter

            with self._joint_lock:
                self._store_joint_sample_locked(
                    *parsed,
                    received_monotonic=received_monotonic,
                    sequence=counter,
                )
                self._joint_worker_last_counter = counter

            # A latest-value slot may skip samples between main-process polls.
            # The public counter preserves exact reception metrics in O(1).
            if self._metrics:
                self._metrics.record_joint_states(metric_delta)

    def get_observation(
        self,
        camera_names: list[str],
    ) -> dict[str, torch.Tensor] | None:
        """Get observation from shared memory if complete."""
        self._refresh_joint_state_from_worker()

        # Every policy observation must contain a new joint state as well as one
        # new frame from every camera. This prevents a fast camera set from
        # repeatedly pairing with a frozen robot state.
        with self._joint_lock:
            if self._joint_sequence <= self._last_consumed_joint_sequence:
                self._last_incomplete_reason = "waiting for a new joint state"
                return None
            joint_sequence = self._joint_sequence
            joint_positions = dict(self._joint_positions or {})
            joint_velocities = (
                dict(self._joint_velocities) if self._joint_velocities is not None else None
            )
            joint_efforts = dict(self._joint_efforts) if self._joint_efforts is not None else None
            joint_received_monotonic = self._joint_received_monotonic
            joint_errors = self._joint_errors

        if joint_received_monotonic is None or not math.isfinite(joint_received_monotonic):
            self._last_incomplete_reason = "joint state has no valid receive timestamp"
            return None

        # Check for complete observation from shared memory.
        images = self._image_buffer.read_all_if_ready_with_metadata()

        if images is None:
            # Not all cameras have new frames yet
            missing = []
            for name in camera_names:
                if not self._image_buffer.has_new_frame(name):
                    missing.append(name)
            self._last_incomplete_reason = f"waiting for cameras: {missing}"
            return None

        # Validate the exact joint sample copied above. A newer callback may
        # have replaced the live fields while the camera snapshot was read; it
        # must not make an invalid sample used by this observation appear valid.
        if joint_errors:
            raise ValueError(
                f"joint sample {joint_sequence} is invalid: " + "; ".join(joint_errors)
            )

        if not joint_positions:
            self._last_incomplete_reason = "waiting for joint state"
            return None

        expected_cameras = set(camera_names)
        consumed_cameras = set(images)
        if consumed_cameras != expected_cameras:
            missing = sorted(expected_cameras - consumed_cameras)
            unexpected = sorted(consumed_cameras - expected_cameras)
            raise ValueError(
                "camera snapshot does not match requested observation "
                f"(missing={missing}, unexpected={unexpected})"
            )

        camera_received_by_name = {
            camera_name: received_monotonic
            for camera_name, (
                _image,
                _timestamp,
                _counter,
                received_monotonic,
            ) in images.items()
        }
        invalid_camera_timestamps = sorted(
            name
            for name, received in camera_received_by_name.items()
            if not math.isfinite(received)
        )
        if invalid_camera_timestamps:
            raise ValueError(
                "camera observation has a non-finite receive timestamp: "
                + ", ".join(invalid_camera_timestamps)
            )

        exact_receipts = {
            "joint_states": joint_received_monotonic,
            **{f"camera:{name}": received for name, received in camera_received_by_name.items()},
        }
        oldest_name, oldest_received = min(exact_receipts.items(), key=lambda item: item[1])
        newest_name, newest_received = max(exact_receipts.items(), key=lambda item: item[1])
        sensor_skew = newest_received - oldest_received
        if sensor_skew > self._max_sensor_skew_sec:
            raise ValueError(
                f"exact sensor receive skew {sensor_skew:.3f}s exceeds "
                f"{self._max_sensor_skew_sec:.3f}s "
                f"(oldest={oldest_name}, newest={newest_name})"
            )

        # Build observation dict
        observation = self._build_observation(
            images,
            joint_positions,
            joint_velocities,
            joint_efforts,
        )
        self._last_observation_sequence = ObservationSequence(
            joint_state=joint_sequence,
            cameras=tuple(
                sorted(
                    (camera_name, frame_counter)
                    for camera_name, (
                        _image,
                        _timestamp,
                        frame_counter,
                        _received_monotonic,
                    ) in images.items()
                )
            ),
        )
        self._last_observation_monotonic = min(
            [joint_received_monotonic, *camera_received_by_name.values()]
        )
        self._last_consumed_joint_sequence = joint_sequence
        self._last_incomplete_reason = ""
        return observation

    def _build_observation(
        self,
        images: dict[str, tuple],
        joint_positions: dict[str, float],
        joint_velocities: dict[str, float] | None,
        joint_efforts: dict[str, float] | None,
    ) -> dict[str, torch.Tensor]:
        """Build observation dict from shared memory images and joint state."""
        observation = {}

        # Add images (already decompressed by workers)
        for camera_name, (
            image,
            _timestamp,
            _frame_counter,
            _received_monotonic,
        ) in images.items():
            # Convert to tensor and normalize to [0, 1]
            image_tensor = torch.from_numpy(image).float() / 255.0
            # Rearrange to (C, H, W) and add batch dimension
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            observation[f"observation.images.{camera_name}"] = image_tensor

        # Build state observations (position / velocity / effort) based on config
        if joint_positions:
            obs_prefix = self._joint_names_config.get("observation_prefix", "follower")
            sep = self._joint_names_config.get("separator", "_")
            arm_mapping = self._joint_names_config.get("arm_mapping", {"l": "left", "r": "right"})
            joint_order = self._joint_names_config.get("model_joint_order", [])
            state_features = self._joint_names_config.get("state_features", ["position"])

            feature_map = {
                "position": (joint_positions, "observation.state"),
                "velocity": (joint_velocities, "observation.velocity"),
                "effort": (joint_efforts, "observation.effort"),
            }

            for feature in state_features:
                if feature not in feature_map:
                    continue
                data_dict, obs_key = feature_map[feature]
                ordered = []
                for arm_key in sorted(arm_mapping.keys()):
                    for joint_id in joint_order:
                        joint_name = f"{obs_prefix}{sep}{arm_key}{sep}{joint_id}"
                        if data_dict is None or joint_name not in data_dict:
                            raise ValueError(f"{feature} missing required joint '{joint_name}'")
                        val = data_dict[joint_name]
                        if not math.isfinite(val):
                            raise ValueError(
                                f"{feature} is non-finite for required joint '{joint_name}'"
                            )
                        ordered.append(val)
                observation[obs_key] = torch.tensor(ordered, dtype=torch.float32).unsqueeze(0)

        return observation

    def get_current_joint_positions(self) -> dict[str, float]:
        """Get current joint positions for delta limiting."""
        self._refresh_joint_state_from_worker()
        with self._joint_lock:
            return dict(self._joint_positions or {})

    def get_last_observation_sequence(self) -> ObservationSequence | None:
        """Return the sequence identity of the last observation built."""
        return self._last_observation_sequence

    def get_last_observation_monotonic(self) -> float | None:
        """Return the oldest exact input receipt time in the last observation."""
        return self._last_observation_monotonic

    def get_input_snapshot(self, camera_names: list[str]) -> InputSnapshot:
        """Return freshness metadata without consuming any sensor sample."""
        metadata = self._image_buffer.get_frame_metadata() if self._image_buffer else {}
        cameras = tuple(
            SensorReading(
                name=f"camera:{name}",
                sequence=metadata.get(name, (0, None))[0],
                last_seen_monotonic=metadata.get(name, (0, None))[1],
            )
            for name in sorted(camera_names)
        )

        self._refresh_joint_state_from_worker()
        with self._joint_lock:
            positions = dict(self._joint_positions or {})
            joint_errors = self._joint_errors
            joint_reading = SensorReading(
                name="joint_states",
                sequence=self._joint_sequence,
                last_seen_monotonic=self._joint_received_monotonic,
            )

        missing_joints = tuple(name for name in self._required_joint_names if name not in positions)
        invalid_joints = tuple(
            name
            for name in self._required_joint_names
            if name in positions and not math.isfinite(positions[name])
        )
        return InputSnapshot(
            joint_state=joint_reading,
            cameras=cameras,
            missing_joints=missing_joints,
            invalid_joints=invalid_joints,
            joint_errors=joint_errors,
        )

    def get_incomplete_reason(self) -> str:
        """Get reason why observation is incomplete."""
        return self._last_incomplete_reason

    def record_metrics(self, metrics_tracker: Any) -> None:
        """Record metrics - joint state is tracked via callback."""
        # Joint state metrics are recorded by main node
        # Image metrics tracked per-camera from shared memory frame counters
        pass

    def get_frame_counters(self) -> dict[str, int]:
        """Get frame counters from shared memory (for stats logging)."""
        if self._image_buffer:
            return self._image_buffer.get_frame_counters()
        return {}

    def cleanup(self) -> None:
        """Stop workers and clean up shared memory."""
        if self._node:
            self._node.get_logger().info("Stopping worker processes...")

        # Signal workers to stop
        if self._stop_event:
            self._stop_event.set()

        # Wait for workers to finish
        for p in self._worker_processes:
            p.join(timeout=2.0)
            if p.is_alive():
                if self._node:
                    self._node.get_logger().warn(f"Force terminating worker {p.name}")
                p.terminate()
                p.join(timeout=1.0)

        if self._node:
            self._node.get_logger().info("All workers stopped")

        # Clean up shared memory
        if self._image_buffer:
            self._image_buffer.unlink()
            self._image_buffer = None
        if self._joint_buffer:
            self._joint_buffer.unlink()
            self._joint_buffer = None
        self._joint_worker_process = None
        self._worker_processes.clear()
        self._stop_event = None
