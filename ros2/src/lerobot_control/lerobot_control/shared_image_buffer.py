"""
Shared Memory Image Buffer for Multi-Process Inference

Provides zero-copy shared memory communication between image worker processes
and the main inference process. Each camera has its own buffer slot with:
- Image data (numpy array)
- ROS timestamp
- Local monotonic receive timestamp
- Ready flag (indicates new data available)
"""

import ctypes
import ctypes.util
import struct
import time
from contextlib import suppress
from multiprocessing import shared_memory

import numpy as np


class SharedImageBuffer:
    """
    Zero-copy shared memory buffer for camera images.

    Memory layout per camera:
    - Image data: H x W x 3 uint8 (e.g., 480 x 640 x 3 = 921,600 bytes)
    - Alignment padding: 0-7 bytes
    - Seqlock version: uint64 (8 bytes) - odd while a write is in progress
    - ROS timestamp: float64 (8 bytes)
    - Local monotonic receive timestamp: float64 (8 bytes)
    - Frame counter: uint64 (8 bytes) - incremented on each write
    - Total per camera: aligned(image_size, 8) + 32 bytes

    Usage:
        # In main process (creates shared memory)
        buffer = SharedImageBuffer(
            camera_names=['waist', 'wrist_r', 'chest', 'wrist_l'],
            image_shape=(480, 640, 3),
            create=True
        )

        # In worker process (attaches to existing shared memory)
        buffer = SharedImageBuffer(
            camera_names=['waist', 'wrist_r', 'chest', 'wrist_l'],
            image_shape=(480, 640, 3),
            create=False
        )
    """

    # Constants for memory layout. All metadata fields are naturally aligned so
    # the version can be loaded/stored atomically by libatomic.
    METADATA_ALIGNMENT = 8
    SEQUENCE_SIZE = 8  # uint64, internal seqlock version
    TIMESTAMP_SIZE = 8  # float64
    MONOTONIC_SIZE = 8  # float64
    COUNTER_SIZE = 8  # uint64, public frame counter
    METADATA_SIZE = SEQUENCE_SIZE + TIMESTAMP_SIZE + MONOTONIC_SIZE + COUNTER_SIZE
    MAX_SNAPSHOT_RETRIES = 100
    SNAPSHOT_RETRY_DELAY_SECONDS = 0.0001
    _MAX_UINT64 = (1 << 64) - 1
    _ATOMIC_SEQ_CST = 5

    def __init__(
        self,
        camera_names: list[str],
        image_shape: tuple[int, int, int],
        create: bool = True,
        buffer_name_prefix: str = "lerobot_img_",
    ):
        """
        Initialize shared image buffer.

        Args:
            camera_names: List of camera names (e.g., ['waist', 'wrist_r', ...])
            image_shape: Shape of images (H, W, C), e.g., (480, 640, 3)
            create: If True, create new shared memory. If False, attach to existing.
            buffer_name_prefix: Prefix for shared memory names
        """
        self.camera_names = camera_names
        self.image_shape = image_shape
        self.image_size = int(np.prod(image_shape))
        self.metadata_offset = self._align_up(self.image_size, self.METADATA_ALIGNMENT)
        self._sequence_offset = self.metadata_offset
        self._timestamp_offset = self._sequence_offset + self.SEQUENCE_SIZE
        self._monotonic_offset = self._timestamp_offset + self.TIMESTAMP_SIZE
        self._counter_offset = self._monotonic_offset + self.MONOTONIC_SIZE
        self.buffer_size = self.metadata_offset + self.METADATA_SIZE
        self.buffer_name_prefix = buffer_name_prefix
        self.create = create

        # Define cleanup-owned state before loading runtime support so a failed
        # constructor remains safe to finalize.
        self._shm_blocks: dict[str, shared_memory.SharedMemory] = {}
        self._atomic_load_8, self._atomic_store_8 = self._load_atomic_operations()

        # Last read frame counters (to detect new frames)
        self._last_read_counters: dict[str, int] = dict.fromkeys(camera_names, 0)

        # Initialize shared memory
        self._init_shared_memory()

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        """Return the next address offset aligned to ``alignment`` bytes."""
        return (value + alignment - 1) & ~(alignment - 1)

    @classmethod
    def _load_atomic_operations(cls):
        """Load acquire/release-capable uint64 operations used by the seqlock."""
        library_name = ctypes.util.find_library("atomic")
        if library_name is None:
            raise RuntimeError(
                "SharedImageBuffer requires libatomic for coherent cross-process snapshots"
            )

        library = ctypes.CDLL(library_name)
        try:
            atomic_load = getattr(library, "__atomic_load_8")
            atomic_store = getattr(library, "__atomic_store_8")
        except AttributeError as exc:
            raise RuntimeError(
                "SharedImageBuffer requires 64-bit atomic load/store support"
            ) from exc

        atomic_load.argtypes = [ctypes.c_void_p, ctypes.c_int]
        atomic_load.restype = ctypes.c_uint64
        atomic_store.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
        atomic_store.restype = None
        # Keep the library alive through the bound function objects.
        atomic_load._shared_image_buffer_library = library
        atomic_store._shared_image_buffer_library = library
        return atomic_load, atomic_store

    def _atomic_load_uint64(self, buf: memoryview, offset: int) -> int:
        """Atomically load an aligned uint64 with sequential consistency."""
        field = ctypes.c_uint64.from_buffer(buf, offset)
        try:
            return int(self._atomic_load_8(ctypes.byref(field), self._ATOMIC_SEQ_CST))
        finally:
            del field

    def _atomic_store_uint64(self, buf: memoryview, offset: int, value: int) -> None:
        """Atomically store an aligned uint64 with sequential consistency."""
        field = ctypes.c_uint64.from_buffer(buf, offset)
        try:
            self._atomic_store_8(ctypes.byref(field), value, self._ATOMIC_SEQ_CST)
        finally:
            del field

    def _get_shm_name(self, camera_name: str) -> str:
        """Get shared memory name for a camera."""
        return f"{self.buffer_name_prefix}{camera_name}"

    def _init_shared_memory(self):
        """Initialize shared memory blocks for all cameras."""
        for camera_name in self.camera_names:
            shm_name = self._get_shm_name(camera_name)

            if self.create:
                # Clean up any existing shared memory with same name
                try:
                    existing = shared_memory.SharedMemory(name=shm_name)
                    existing.close()
                    existing.unlink()
                except FileNotFoundError:
                    pass

                # Create new shared memory
                shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.buffer_size)
                # Initialize to zeros
                np.ndarray((self.buffer_size,), dtype=np.uint8, buffer=shm.buf).fill(0)
            else:
                # Attach to existing shared memory (with retry for startup timing)
                max_retries = 50
                for i in range(max_retries):
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name)
                        break
                    except FileNotFoundError:
                        if i < max_retries - 1:
                            time.sleep(0.1)
                        else:
                            raise RuntimeError(
                                f"Shared memory '{shm_name}' not found. "
                                "Make sure the main process creates it first."
                            ) from None

                if shm.size != self.buffer_size:
                    actual_size = shm.size
                    shm.close()
                    raise RuntimeError(
                        f"Shared memory '{shm_name}' has incompatible size {actual_size}; "
                        f"expected {self.buffer_size}. Ensure every process uses the same "
                        "SharedImageBuffer layout and image shape."
                    )

            self._shm_blocks[camera_name] = shm

    def write(
        self,
        camera_name: str,
        image: np.ndarray,
        timestamp: float,
        received_monotonic: float | None = None,
    ):
        """
        Write image to shared memory buffer.

        Called by image worker process after decompressing JPEG.

        Args:
            camera_name: Name of the camera
            image: Decompressed image as numpy array (H, W, 3) uint8
            timestamp: ROS2 message timestamp, retained for synchronization diagnostics
            received_monotonic: Local receipt time. Defaults to ``time.monotonic()``.
        """
        if camera_name not in self._shm_blocks:
            raise ValueError(f"Unknown camera: {camera_name}")

        # Validate image shape
        if image.shape != self.image_shape:
            raise ValueError(f"Image shape {image.shape} doesn't match expected {self.image_shape}")

        # Prepare everything that can fail before making the seqlock odd. An
        # exception after that point intentionally leaves the version odd, so
        # readers fail closed rather than accepting a partial frame.
        image_bytes = np.ascontiguousarray(image).reshape(-1).tobytes()
        if len(image_bytes) != self.image_size:
            raise ValueError("Image data must contain one uint8 byte per image element")
        timestamp_bytes = struct.pack("=d", timestamp)
        monotonic_bytes = struct.pack(
            "=d", time.monotonic() if received_monotonic is None else received_monotonic
        )

        buf = self._shm_blocks[camera_name].buf
        sequence = self._atomic_load_uint64(buf, self._sequence_offset)
        if sequence & 1:
            raise RuntimeError(
                f"Cannot write camera '{camera_name}': its shared-memory seqlock is already odd"
            )
        if sequence > self._MAX_UINT64 - 2:
            raise OverflowError(f"Seqlock version exhausted for camera '{camera_name}'")

        current_counter = struct.unpack_from("=Q", buf, self._counter_offset)[0]
        if current_counter == self._MAX_UINT64:
            raise OverflowError(f"Frame counter exhausted for camera '{camera_name}'")
        new_counter = current_counter + 1
        counter_bytes = struct.pack("=Q", new_counter)

        # A single worker writes each camera. Odd marks an update in progress;
        # the final even store publishes image and metadata as one snapshot.
        self._atomic_store_uint64(buf, self._sequence_offset, sequence + 1)
        buf[: self.image_size] = image_bytes
        buf[self._timestamp_offset : self._timestamp_offset + self.TIMESTAMP_SIZE] = timestamp_bytes
        buf[self._monotonic_offset : self._monotonic_offset + self.MONOTONIC_SIZE] = monotonic_bytes
        buf[self._counter_offset : self._counter_offset + self.COUNTER_SIZE] = counter_bytes
        self._atomic_store_uint64(buf, self._sequence_offset, sequence + 2)

    def _read_consistent_snapshot(
        self,
        camera_name: str,
        *,
        copy_image: bool,
    ) -> tuple[bytes | None, float, float, int]:
        """Read one coherent frame snapshot or fail after bounded retries."""
        if camera_name not in self._shm_blocks:
            raise ValueError(f"Unknown camera: {camera_name}")

        buf = self._shm_blocks[camera_name].buf
        version_before = 0
        version_after = 0
        for attempt in range(self.MAX_SNAPSHOT_RETRIES):
            version_before = self._atomic_load_uint64(buf, self._sequence_offset)
            if version_before & 1:
                version_after = version_before
                if attempt + 1 < self.MAX_SNAPSHOT_RETRIES:
                    time.sleep(self.SNAPSHOT_RETRY_DELAY_SECONDS)
                continue

            image_data = bytes(buf[: self.image_size]) if copy_image else None
            timestamp = struct.unpack_from("=d", buf, self._timestamp_offset)[0]
            received_monotonic = struct.unpack_from("=d", buf, self._monotonic_offset)[0]
            frame_counter = struct.unpack_from("=Q", buf, self._counter_offset)[0]

            version_after = self._atomic_load_uint64(buf, self._sequence_offset)
            if version_before == version_after and not (version_after & 1):
                return image_data, timestamp, received_monotonic, frame_counter

            if attempt + 1 < self.MAX_SNAPSHOT_RETRIES:
                time.sleep(self.SNAPSHOT_RETRY_DELAY_SECONDS)

        raise RuntimeError(
            f"Could not obtain a coherent shared-memory snapshot for camera '{camera_name}' "
            f"after {self.MAX_SNAPSHOT_RETRIES} attempts "
            f"(versions {version_before} -> {version_after})"
        )

    def read(self, camera_name: str) -> tuple[np.ndarray, float, int]:
        """
        Read image from shared memory buffer.

        Args:
            camera_name: Name of the camera

        Returns:
            Tuple of (image, timestamp, frame_counter)
        """
        image_data, timestamp, _received_monotonic, frame_counter = self._read_consistent_snapshot(
            camera_name, copy_image=True
        )
        if image_data is None:  # Defensive: copy_image=True always returns bytes.
            raise RuntimeError(f"Missing image data for camera '{camera_name}'")
        image = np.frombuffer(image_data, dtype=np.uint8).reshape(self.image_shape).copy()
        return image, timestamp, frame_counter

    def has_new_frame(self, camera_name: str) -> bool:
        """Check if camera has a new frame since last read."""
        if camera_name not in self._shm_blocks:
            return False
        _image, _timestamp, _received_monotonic, current_counter = self._read_consistent_snapshot(
            camera_name, copy_image=False
        )
        return current_counter > self._last_read_counters[camera_name]

    def read_if_new(self, camera_name: str) -> tuple[np.ndarray, float] | None:
        """
        Read image only if there's a new frame.

        Returns:
            Tuple of (image, timestamp) if new frame available, None otherwise
        """
        if camera_name not in self._shm_blocks:
            return None

        image, timestamp, frame_counter = self.read(camera_name)
        if frame_counter <= self._last_read_counters[camera_name]:
            return None
        self._last_read_counters[camera_name] = frame_counter
        return image, timestamp

    def _read_all_new_snapshots(
        self,
    ) -> dict[str, tuple[np.ndarray, float, int, float]] | None:
        """Read every camera once, retaining the metadata of each exact frame."""
        snapshots: dict[str, tuple[np.ndarray, float, int, float]] = {}
        for camera_name in self.camera_names:
            image_data, timestamp, received_monotonic, frame_counter = (
                self._read_consistent_snapshot(camera_name, copy_image=True)
            )
            if image_data is None:  # Defensive: copy_image=True always returns bytes.
                raise RuntimeError(f"Missing image data for camera '{camera_name}'")
            if frame_counter <= self._last_read_counters[camera_name]:
                return None
            image = np.frombuffer(image_data, dtype=np.uint8).reshape(self.image_shape).copy()
            snapshots[camera_name] = (
                image,
                timestamp,
                frame_counter,
                received_monotonic,
            )

        for camera_name, (_image, _timestamp, frame_counter, _received) in snapshots.items():
            self._last_read_counters[camera_name] = frame_counter
        return snapshots

    def read_all_if_ready(self) -> dict[str, tuple[np.ndarray, float]] | None:
        """
        Read all cameras only if ALL have new frames.

        This ensures synchronized observations across all cameras.

        Returns:
            Dict mapping camera_name -> (image, timestamp) if all ready, None otherwise
        """
        snapshots = self._read_all_new_snapshots()
        if snapshots is None:
            return None
        return {
            camera_name: (image, timestamp)
            for camera_name, (
                image,
                timestamp,
                _frame_counter,
                _received_monotonic,
            ) in snapshots.items()
        }

    def read_all_if_ready_with_counters(
        self,
    ) -> dict[str, tuple[np.ndarray, float, int]] | None:
        """Read one new frame per camera and retain each frame sequence.

        This legacy API deliberately keeps its original three-item tuple. New
        observation code should use :meth:`read_all_if_ready_with_metadata` so
        action provenance is tied to the exact consumed camera frames.
        """
        snapshots = self._read_all_new_snapshots()
        if snapshots is None:
            return None
        return {
            camera_name: (image, timestamp, frame_counter)
            for camera_name, (
                image,
                timestamp,
                frame_counter,
                _received_monotonic,
            ) in snapshots.items()
        }

    def read_all_if_ready_with_metadata(
        self,
    ) -> dict[str, tuple[np.ndarray, float, int, float]] | None:
        """Read new frames with their exact counter and monotonic receipt time.

        Tuple fields are ``(image, ros_timestamp, frame_counter,
        received_monotonic)``. Metadata is copied under the same per-camera
        seqlock snapshot as the image, so a later writer cannot change the
        provenance of the returned observation.
        """
        return self._read_all_new_snapshots()

    def get_frame_metadata(self) -> dict[str, tuple[int, float | None]]:
        """Return ``(sequence, received_monotonic)`` for every camera."""
        metadata: dict[str, tuple[int, float | None]] = {}
        for camera_name in self.camera_names:
            _image, _timestamp, received_monotonic, counter = self._read_consistent_snapshot(
                camera_name, copy_image=False
            )
            metadata[camera_name] = (
                counter,
                received_monotonic if counter > 0 else None,
            )
        return metadata

    def get_frame_counters(self) -> dict[str, int]:
        """Get current frame counters for all cameras."""
        counters = {}
        for camera_name in self.camera_names:
            _image, _timestamp, _received_monotonic, counter = self._read_consistent_snapshot(
                camera_name, copy_image=False
            )
            counters[camera_name] = counter
        return counters

    def close(self):
        """Close shared memory connections."""
        for shm in self._shm_blocks.values():
            with suppress(Exception):
                shm.close()

    def unlink(self):
        """Unlink (delete) shared memory blocks. Only call from creating process."""
        for shm in self._shm_blocks.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._shm_blocks.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class SharedJointStateBuffer:
    """Coherent shared-memory slot for one serialized ROS ``JointState``.

    The worker stores the complete CDR payload instead of a filtered positions
    array. The main process can therefore apply the same parser and validation
    used by its in-process subscription, including malformed-array diagnostics,
    velocity, effort, the ROS header timestamp, and the original joint names.

    Memory layout:
    - Serialized CDR payload: fixed-capacity byte array
    - Alignment padding: 0-7 bytes
    - Seqlock version: uint64 (odd while a write is in progress)
    - Serialized payload length: uint64
    - Local monotonic receive timestamp: float64
    - Public message counter: uint64
    """

    DEFAULT_PAYLOAD_CAPACITY = 64 * 1024
    METADATA_ALIGNMENT = SharedImageBuffer.METADATA_ALIGNMENT
    SEQUENCE_SIZE = SharedImageBuffer.SEQUENCE_SIZE
    LENGTH_SIZE = 8
    MONOTONIC_SIZE = SharedImageBuffer.MONOTONIC_SIZE
    COUNTER_SIZE = SharedImageBuffer.COUNTER_SIZE
    METADATA_SIZE = SEQUENCE_SIZE + LENGTH_SIZE + MONOTONIC_SIZE + COUNTER_SIZE
    MAX_SNAPSHOT_RETRIES = SharedImageBuffer.MAX_SNAPSHOT_RETRIES
    SNAPSHOT_RETRY_DELAY_SECONDS = SharedImageBuffer.SNAPSHOT_RETRY_DELAY_SECONDS
    _MAX_UINT64 = SharedImageBuffer._MAX_UINT64
    _ATOMIC_SEQ_CST = SharedImageBuffer._ATOMIC_SEQ_CST

    def __init__(
        self,
        create: bool = True,
        buffer_name: str = "lerobot_joint_state",
        payload_capacity: int = DEFAULT_PAYLOAD_CAPACITY,
    ):
        if isinstance(payload_capacity, bool) or not isinstance(payload_capacity, int):
            raise TypeError("payload_capacity must be an integer")
        if payload_capacity <= 0:
            raise ValueError("payload_capacity must be > 0")
        if not isinstance(buffer_name, str) or not buffer_name:
            raise ValueError("buffer_name must be a non-empty string")

        self.payload_capacity = payload_capacity
        self.metadata_offset = SharedImageBuffer._align_up(
            payload_capacity,
            self.METADATA_ALIGNMENT,
        )
        self._sequence_offset = self.metadata_offset
        self._length_offset = self._sequence_offset + self.SEQUENCE_SIZE
        self._monotonic_offset = self._length_offset + self.LENGTH_SIZE
        self._counter_offset = self._monotonic_offset + self.MONOTONIC_SIZE
        self.buffer_size = self.metadata_offset + self.METADATA_SIZE
        self.buffer_name = buffer_name
        self.create = create
        self._last_read_counter = 0
        self._shm: shared_memory.SharedMemory | None = None
        self._atomic_load_8, self._atomic_store_8 = SharedImageBuffer._load_atomic_operations()

        self._init_shared_memory()

    def _atomic_load_uint64(self, buf: memoryview, offset: int) -> int:
        """Atomically load an aligned uint64 with sequential consistency."""
        field = ctypes.c_uint64.from_buffer(buf, offset)
        try:
            return int(self._atomic_load_8(ctypes.byref(field), self._ATOMIC_SEQ_CST))
        finally:
            del field

    def _atomic_store_uint64(self, buf: memoryview, offset: int, value: int) -> None:
        """Atomically store an aligned uint64 with sequential consistency."""
        field = ctypes.c_uint64.from_buffer(buf, offset)
        try:
            self._atomic_store_8(ctypes.byref(field), value, self._ATOMIC_SEQ_CST)
        finally:
            del field

    def _init_shared_memory(self) -> None:
        """Create or attach to the single joint-state slot."""
        if self.create:
            try:
                existing = shared_memory.SharedMemory(name=self.buffer_name)
                existing.close()
                existing.unlink()
            except FileNotFoundError:
                pass

            self._shm = shared_memory.SharedMemory(
                name=self.buffer_name,
                create=True,
                size=self.buffer_size,
            )
            np.ndarray((self.buffer_size,), dtype=np.uint8, buffer=self._shm.buf).fill(0)
            return

        max_retries = 50
        for attempt in range(max_retries):
            try:
                self._shm = shared_memory.SharedMemory(name=self.buffer_name)
                break
            except FileNotFoundError:
                if attempt + 1 < max_retries:
                    time.sleep(0.1)
                    continue
                raise RuntimeError(
                    f"Shared memory '{self.buffer_name}' not found. "
                    "Make sure the main process creates it first."
                ) from None

        if self._shm is None:  # Defensive: the retry loop either attaches or raises.
            raise RuntimeError(f"Could not attach shared memory '{self.buffer_name}'")
        if self._shm.size != self.buffer_size:
            actual_size = self._shm.size
            self._shm.close()
            self._shm = None
            raise RuntimeError(
                f"Shared memory '{self.buffer_name}' has incompatible size {actual_size}; "
                f"expected {self.buffer_size}. Ensure every process uses the same "
                "SharedJointStateBuffer payload capacity."
            )

    def _buffer(self) -> memoryview:
        """Return the live shared-memory view or reject use after construction failure."""
        if self._shm is None:
            raise RuntimeError("Joint-state shared memory is not available")
        return self._shm.buf

    def write(self, serialized_message: bytes, received_monotonic: float) -> int:
        """Publish one serialized ``JointState`` and return its message counter."""
        if not isinstance(serialized_message, (bytes, bytearray, memoryview)):
            raise TypeError("serialized_message must be bytes-like")
        payload = bytes(serialized_message)
        if not payload:
            raise ValueError("serialized_message must not be empty")
        if len(payload) > self.payload_capacity:
            raise ValueError(
                f"serialized JointState is {len(payload)} bytes; "
                f"slot capacity is {self.payload_capacity} bytes"
            )
        if not np.isfinite(received_monotonic):
            raise ValueError("received_monotonic must be finite")

        # Prepare metadata before making the slot odd. Any exception after the
        # odd store intentionally leaves the slot unreadable instead of exposing
        # a partially updated ROS message as a coherent sample.
        length_bytes = struct.pack("=Q", len(payload))
        monotonic_bytes = struct.pack("=d", received_monotonic)
        buf = self._buffer()
        sequence = self._atomic_load_uint64(buf, self._sequence_offset)
        if sequence & 1:
            raise RuntimeError("Cannot write joint state: shared-memory seqlock is already odd")
        if sequence > self._MAX_UINT64 - 2:
            raise OverflowError("Joint-state seqlock version exhausted")

        current_counter = struct.unpack_from("=Q", buf, self._counter_offset)[0]
        if current_counter == self._MAX_UINT64:
            raise OverflowError("Joint-state message counter exhausted")
        new_counter = current_counter + 1
        counter_bytes = struct.pack("=Q", new_counter)

        self._atomic_store_uint64(buf, self._sequence_offset, sequence + 1)
        buf[: len(payload)] = payload
        buf[self._length_offset : self._length_offset + self.LENGTH_SIZE] = length_bytes
        buf[self._monotonic_offset : self._monotonic_offset + self.MONOTONIC_SIZE] = (
            monotonic_bytes
        )
        buf[self._counter_offset : self._counter_offset + self.COUNTER_SIZE] = counter_bytes
        self._atomic_store_uint64(buf, self._sequence_offset, sequence + 2)
        return new_counter

    def _read_consistent_snapshot(
        self,
        *,
        copy_payload: bool,
    ) -> tuple[bytes | None, float | None, int]:
        """Read one coherent slot value or fail after bounded retries."""
        buf = self._buffer()
        version_before = 0
        version_after = 0
        for attempt in range(self.MAX_SNAPSHOT_RETRIES):
            version_before = self._atomic_load_uint64(buf, self._sequence_offset)
            if version_before & 1:
                version_after = version_before
                if attempt + 1 < self.MAX_SNAPSHOT_RETRIES:
                    time.sleep(self.SNAPSHOT_RETRY_DELAY_SECONDS)
                continue

            payload_length = struct.unpack_from("=Q", buf, self._length_offset)[0]
            received_monotonic = struct.unpack_from("=d", buf, self._monotonic_offset)[0]
            counter = struct.unpack_from("=Q", buf, self._counter_offset)[0]
            payload = bytes(buf[:payload_length]) if copy_payload and payload_length else None

            version_after = self._atomic_load_uint64(buf, self._sequence_offset)
            if version_before != version_after or version_after & 1:
                if attempt + 1 < self.MAX_SNAPSHOT_RETRIES:
                    time.sleep(self.SNAPSHOT_RETRY_DELAY_SECONDS)
                continue

            if payload_length > self.payload_capacity:
                raise RuntimeError(
                    f"Joint-state shared-memory payload length {payload_length} exceeds "
                    f"slot capacity {self.payload_capacity}"
                )
            if counter == 0:
                if payload_length != 0:
                    raise RuntimeError(
                        "Joint-state shared memory has a payload without a message counter"
                    )
                return (b"" if copy_payload else None), None, 0
            if payload_length == 0:
                raise RuntimeError(
                    "Joint-state shared memory has a message counter without a payload"
                )
            if not np.isfinite(received_monotonic):
                raise RuntimeError(
                    "Joint-state shared memory has a non-finite receive timestamp"
                )
            if copy_payload and payload is None:  # Defensive: length is positive here.
                raise RuntimeError("Joint-state shared memory payload could not be copied")
            return payload, received_monotonic, counter

        raise RuntimeError(
            "Could not obtain a coherent shared-memory snapshot for joint state "
            f"after {self.MAX_SNAPSHOT_RETRIES} attempts "
            f"(versions {version_before} -> {version_after})"
        )

    def read(self) -> tuple[bytes, float | None, int]:
        """Return the exact serialized payload, receive time, and public counter."""
        payload, received_monotonic, counter = self._read_consistent_snapshot(copy_payload=True)
        if payload is None:  # Defensive: copy_payload=True always returns bytes.
            raise RuntimeError("Joint-state shared memory did not return a payload")
        return payload, received_monotonic, counter

    def read_if_new(self) -> tuple[bytes, float, int] | None:
        """Return the exact slot value only if its public counter advanced."""
        payload, received_monotonic, counter = self.read()
        if counter <= self._last_read_counter:
            return None
        if received_monotonic is None:  # A positive counter always has a timestamp.
            raise RuntimeError("Joint-state shared memory is missing its receive timestamp")
        self._last_read_counter = counter
        return payload, received_monotonic, counter

    def get_metadata(self) -> tuple[int, float | None]:
        """Return the public message counter and exact receive timestamp."""
        _payload, received_monotonic, counter = self._read_consistent_snapshot(
            copy_payload=False
        )
        return counter, received_monotonic

    def close(self) -> None:
        if self._shm is not None:
            with suppress(Exception):
                self._shm.close()

    def unlink(self) -> None:
        if self._shm is None:
            return
        with suppress(FileNotFoundError):
            self._shm.unlink()
        with suppress(Exception):
            self._shm.close()
        self._shm = None

    def __del__(self):
        """Close this process's handle without unlinking an owned slot."""
        self.close()
