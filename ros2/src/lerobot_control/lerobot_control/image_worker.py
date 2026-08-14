"""
Image Worker Process for Multi-Process Inference

Each image worker runs in a separate process, subscribing to a single camera topic,
decompressing JPEG images, and writing to shared memory. This eliminates GIL
contention with the main inference process.
"""

import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CompressedImage, JointState

from .jpeg_integrity import JpegDecodeResult, StrictJpegDecoder
from .shared_image_buffer import SharedImageBuffer, SharedJointStateBuffer

BAD_FRAME_LOG_INTERVAL_SECONDS = 5.0


def joint_state_qos_profile() -> QoSProfile:
    """Return the QoS contract shared by both joint-state ingress modes."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


class ImageWorkerNode(Node):
    """
    ROS2 node that subscribes to a single camera and writes to shared memory.

    Runs in its own process for true parallelism (no GIL).
    """

    def __init__(
        self,
        camera_topic: str,
        camera_name: str,
        image_shape: tuple[int, int, int],
        buffer_name_prefix: str = "lerobot_img_",
        debug_dir: str | None = None,
        debug_max_frames: int = 10,
    ):
        super().__init__(f"image_worker_{camera_name}")

        self.camera_name = camera_name
        self.camera_topic = camera_topic
        self.image_shape = image_shape

        self._debug_dir = Path(debug_dir) / camera_name if debug_dir else None
        self._debug_max_frames = debug_max_frames
        self._debug_saved = 0
        self._debug_last_save: float = 0.0

        # Connect to shared memory (created by main process)
        self.shared_buffer = SharedImageBuffer(
            camera_names=[camera_name],
            image_shape=image_shape,
            create=False,
            buffer_name_prefix=buffer_name_prefix,
        )

        # Statistics
        self.frame_count = 0
        self.bad_frame_count = 0
        self.decode_error_count = 0
        self.jpeg_warning_count = 0
        self.invalid_jpeg_count = 0
        self._jpeg_decoder = StrictJpegDecoder()
        self._last_bad_frame_log_monotonic: float | None = None
        self._bad_frames_since_log = 0
        self._bad_reasons_since_log: Counter[str] = Counter()
        self._last_bad_detail = ""

        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            CompressedImage, camera_topic, self._image_callback, qos_profile_sensor_data
        )

        self.get_logger().info(f"Image worker started: {camera_topic} -> {camera_name}")

    def _emit_bad_frame_summary(self, now: float) -> None:
        """Log accumulated JPEG rejections without flooding the ROS log."""
        if self._bad_frames_since_log == 0:
            return
        reason_counts = ",".join(
            f"{reason}={count}" for reason, count in sorted(self._bad_reasons_since_log.items())
        )
        detail = f" detail={self._last_bad_detail!r}" if self._last_bad_detail else ""
        self.get_logger().warning(
            f"Rejected JPEG frame(s) from {self.camera_name}: "
            f"since_last_log={self._bad_frames_since_log} reasons=[{reason_counts}] "
            f"bad_total={self.bad_frame_count} decode_errors={self.decode_error_count} "
            f"libjpeg_warnings={self.jpeg_warning_count} "
            f"invalid_markers={self.invalid_jpeg_count}{detail}"
        )
        self._last_bad_frame_log_monotonic = now
        self._bad_frames_since_log = 0
        self._bad_reasons_since_log.clear()
        self._last_bad_detail = ""

    def _reject_bad_frame(self, result: JpegDecodeResult) -> None:
        """Account for a rejected JPEG and emit a rate-limited diagnostic."""
        reason = result.rejection_reason or "unknown_jpeg_error"
        self.bad_frame_count += 1
        self.decode_error_count += int(result.decode_failed)
        self.jpeg_warning_count += int(result.has_native_warning)
        self.invalid_jpeg_count += int(result.marker_error is not None)
        self._bad_frames_since_log += 1
        self._bad_reasons_since_log[reason] += 1

        details = []
        if result.decode_exception:
            details.append(result.decode_exception)
        if result.native_stderr:
            # Retain every captured native line in the rate-limited diagnostic;
            # do not silently filter warning variants from libjpeg.
            details.append(result.native_stderr)
        self._last_bad_detail = " | ".join(details)

        now = time.monotonic()
        if (
            self._last_bad_frame_log_monotonic is None
            or now - self._last_bad_frame_log_monotonic >= BAD_FRAME_LOG_INTERVAL_SECONDS
        ):
            self._emit_bad_frame_summary(now)

    def _image_callback(self, msg: CompressedImage):
        """Process incoming compressed image."""
        # Match the joint-state watchdog semantics: receipt time is when the
        # subscription callback starts, not when JPEG decode/resize finishes.
        # Otherwise CPU work is incorrectly reported as inter-sensor skew.
        received_monotonic = time.monotonic()
        try:
            # Copy the serialized payload before entering native decode. Each
            # worker is a single-threaded process, so its temporary stderr
            # redirection is isolated from the other camera workers.
            payload = bytes(msg.data)
            try:
                decode_result = self._jpeg_decoder.decode(payload)
            except Exception as exc:  # noqa: BLE001 - fail closed on capture/decoder faults.
                decode_result = JpegDecodeResult(
                    image=None,
                    decode_exception=f"{type(exc).__name__}: {exc}",
                )

            if decode_result.rejection_reason is not None:
                self._reject_bad_frame(decode_result)
                return
            image = decode_result.image
            if image is None:  # Defensive: rejection_reason covers this path.
                self._reject_bad_frame(JpegDecodeResult(image=None))
                return

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize with padding to preserve aspect ratio
            if image.shape[:2] != self.image_shape[:2]:
                target_h, target_w = self.image_shape[:2]
                src_h, src_w = image.shape[:2]
                scale = min(target_w / src_w, target_h / src_h)
                new_w = int(src_w * scale)
                new_h = int(src_h * scale)
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                offset_x = (target_w - new_w) // 2
                offset_y = (target_h - new_h) // 2
                canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
                image = canvas

            # Save debug frames at 1 Hz up to debug_max_frames (before model input, uint8 RGB)
            if self._debug_dir is not None and self._debug_saved < self._debug_max_frames:
                now = time.time()
                if now - self._debug_last_save >= 1.0:
                    self._debug_dir.mkdir(parents=True, exist_ok=True)
                    fname = self._debug_dir / f"frame_{self._debug_saved:04d}.png"
                    cv2.imwrite(str(fname), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    self._debug_last_save = now
                    self._debug_saved += 1
                    if self._debug_saved == self._debug_max_frames:
                        self.get_logger().info(
                            f"[debug] Saved {self._debug_max_frames} frames to {self._debug_dir}"
                        )

            # Get timestamp from message
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            # Write to shared memory
            self.shared_buffer.write(
                self.camera_name,
                image,
                timestamp,
                received_monotonic=received_monotonic,
            )

            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def destroy_node(self):
        """Cleanup."""
        if self._bad_frames_since_log:
            self._emit_bad_frame_summary(time.monotonic())
        self._jpeg_decoder.close()
        self.shared_buffer.close()
        super().destroy_node()


def run_image_worker(
    camera_topic: str,
    camera_name: str,
    image_shape: tuple[int, int, int],
    buffer_name_prefix: str = "lerobot_img_",
    stop_event=None,
    debug_dir: str | None = None,
    debug_max_frames: int = 10,
):
    """
    Entry point for running image worker in a separate process.

    Args:
        camera_topic: ROS2 topic to subscribe to
        camera_name: Name of the camera (e.g., 'waist')
        image_shape: Shape of images (H, W, C)
        buffer_name_prefix: Prefix for shared memory names
        stop_event: Optional multiprocessing.Event to signal shutdown
    """
    # This entry point runs in a spawned child process. Limit OpenCV here,
    # rather than at module import time, so importing image_worker from the
    # main inference process does not alter its OpenCV thread pool.
    cv2.setNumThreads(1)
    rclpy.init(args=[])  # Empty args to avoid inheriting parent's --ros-args node name

    node = ImageWorkerNode(
        camera_topic=camera_topic,
        camera_name=camera_name,
        image_shape=image_shape,
        buffer_name_prefix=buffer_name_prefix,
        debug_dir=debug_dir,
        debug_max_frames=debug_max_frames,
    )
    node.get_logger().info(f"OpenCV worker threads: {cv2.getNumThreads()}")

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        if stop_event is not None:
            # Spin with stop check
            while not stop_event.is_set() and rclpy.ok():
                executor.spin_once(timeout_sec=0.01)
        else:
            # Spin forever
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


class JointStateWorkerNode(Node):
    """
    ROS2 node that subscribes to joint states and writes to shared memory.

    The complete serialized ROS message is retained so the main process applies
    the same parsing and validation as the legacy in-process subscription.
    """

    def __init__(
        self,
        joint_topic: str,
        buffer_name: str = "lerobot_joint_state",
        payload_capacity: int = SharedJointStateBuffer.DEFAULT_PAYLOAD_CAPACITY,
    ):
        super().__init__("joint_state_worker")

        # Connect to shared memory
        self.shared_buffer = SharedJointStateBuffer(
            create=False,
            buffer_name=buffer_name,
            payload_capacity=payload_capacity,
        )

        # Match the QoS used by the in-process subscription exactly.
        self.subscription = self.create_subscription(
            JointState,
            joint_topic,
            self._joint_callback,
            joint_state_qos_profile(),
        )

        self.frame_count = 0

        self.get_logger().info(f"Joint state worker started: {joint_topic}")

    def _joint_callback(self, msg: JointState) -> None:
        """Store one complete message with its callback-ingress timestamp."""
        received_monotonic = time.monotonic()
        try:
            payload = serialize_message(msg)
            self.shared_buffer.write(payload, received_monotonic)
            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Error processing joint state: {e}")

    def destroy_node(self):
        self.shared_buffer.close()
        super().destroy_node()


def run_joint_state_worker(
    joint_topic: str,
    buffer_name: str = "lerobot_joint_state",
    payload_capacity: int = SharedJointStateBuffer.DEFAULT_PAYLOAD_CAPACITY,
    stop_event=None,
):
    """Entry point for joint state worker process."""
    rclpy.init(args=[])  # Empty args to avoid inheriting parent's --ros-args node name

    node = JointStateWorkerNode(
        joint_topic=joint_topic,
        buffer_name=buffer_name,
        payload_capacity=payload_capacity,
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        if stop_event is not None:
            while not stop_event.is_set() and rclpy.ok():
                executor.spin_once(timeout_sec=0.01)
        else:
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
