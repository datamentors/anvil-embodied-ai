"""Shared-memory coherence and metadata tests used by the inference watchdog."""

import multiprocessing as mp
import struct
import time
from uuid import uuid4

import numpy as np
import pytest
from lerobot_control.shared_image_buffer import SharedImageBuffer


def _write_correlated_frames(
    prefix: str,
    image_shape: tuple[int, int, int],
    frame_count: int,
    started,
    finished,
) -> None:
    """Write frames whose pixels and timestamps encode the public counter."""
    buffer = SharedImageBuffer(
        camera_names=["camera"],
        image_shape=image_shape,
        create=False,
        buffer_name_prefix=prefix,
    )
    try:
        started.set()
        for counter in range(1, frame_count + 1):
            image = np.full(image_shape, counter % 251, dtype=np.uint8)
            buffer.write(
                "camera",
                image,
                timestamp=counter + 0.25,
                received_monotonic=counter + 1000.5,
            )
            # Keep the processes overlapped long enough to exercise reads
            # during image copies, not only before and after the write loop.
            time.sleep(0.0005)
    finally:
        buffer.close()
        finished.set()


def _leave_write_in_progress(
    prefix: str,
    image_shape: tuple[int, int, int],
    marked_odd,
) -> None:
    """Emulate a writer that exits after beginning, but before publishing, a frame."""
    buffer = SharedImageBuffer(
        camera_names=["camera"],
        image_shape=image_shape,
        create=False,
        buffer_name_prefix=prefix,
    )
    try:
        buf = buffer._shm_blocks["camera"].buf
        sequence = buffer._atomic_load_uint64(buf, buffer._sequence_offset)
        buffer._atomic_store_uint64(buf, buffer._sequence_offset, sequence + 1)
        buf[: buffer.image_size // 2] = bytes([99]) * (buffer.image_size // 2)
        marked_odd.set()
    finally:
        buffer.close()


def test_monotonic_metadata_and_sequences_advance() -> None:
    prefix = f"lerobot_test_{uuid4().hex}_"
    buffer = SharedImageBuffer(
        camera_names=["base", "wrist"],
        image_shape=(2, 3, 3),
        create=True,
        buffer_name_prefix=prefix,
    )
    try:
        empty = buffer.get_frame_metadata()
        assert empty == {"base": (0, None), "wrist": (0, None)}

        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        buffer.write("base", image, timestamp=42.0, received_monotonic=100.0)
        buffer.write("wrist", image, timestamp=42.1, received_monotonic=100.1)

        assert buffer.get_frame_metadata() == {
            "base": (1, 100.0),
            "wrist": (1, 100.1),
        }
        first = buffer.read_all_if_ready_with_counters()
        assert first is not None
        assert first["base"][2] == 1
        assert first["wrist"][2] == 1
        assert buffer.read_all_if_ready_with_counters() is None

        buffer.write("base", image, timestamp=43.0, received_monotonic=101.0)
        assert buffer.read_all_if_ready_with_counters() is None
        buffer.write("wrist", image, timestamp=43.1, received_monotonic=101.1)
        second = buffer.read_all_if_ready_with_counters()
        assert second is not None
        assert second["base"][2] == 2
        assert second["wrist"][2] == 2
    finally:
        buffer.unlink()


def test_all_camera_read_retains_metadata_when_a_previous_camera_advances(
    monkeypatch,
) -> None:
    """A later writer cannot replace metadata of an already copied frame."""
    prefix = f"lerobot_test_{uuid4().hex}_"
    buffer = SharedImageBuffer(
        camera_names=["base", "wrist"],
        image_shape=(2, 2, 3),
        create=True,
        buffer_name_prefix=prefix,
    )
    try:
        base_first = np.full(buffer.image_shape, 1, dtype=np.uint8)
        base_second = np.full(buffer.image_shape, 9, dtype=np.uint8)
        wrist_first = np.full(buffer.image_shape, 2, dtype=np.uint8)
        buffer.write("base", base_first, timestamp=1.0, received_monotonic=10.0)
        buffer.write("wrist", wrist_first, timestamp=2.0, received_monotonic=20.0)

        real_read = buffer._read_consistent_snapshot
        advanced_base = False

        def advance_base_between_camera_reads(camera_name, *, copy_image):
            nonlocal advanced_base
            if camera_name == "wrist" and copy_image and not advanced_base:
                advanced_base = True
                buffer.write(
                    "base",
                    base_second,
                    timestamp=3.0,
                    received_monotonic=100.0,
                )
            return real_read(camera_name, copy_image=copy_image)

        monkeypatch.setattr(
            buffer,
            "_read_consistent_snapshot",
            advance_base_between_camera_reads,
        )

        snapshots = buffer.read_all_if_ready_with_metadata()
        assert snapshots is not None
        base_image, base_timestamp, base_counter, base_received = snapshots["base"]
        wrist_image, wrist_timestamp, wrist_counter, wrist_received = snapshots["wrist"]

        np.testing.assert_array_equal(base_image, base_first)
        np.testing.assert_array_equal(wrist_image, wrist_first)
        assert (base_timestamp, base_counter, base_received) == (1.0, 1, 10.0)
        assert (wrist_timestamp, wrist_counter, wrist_received) == (2.0, 1, 20.0)

        # The live buffer has advanced, proving the returned provenance did not
        # come from a global metadata read after the multi-camera copy.
        assert buffer.get_frame_metadata()["base"] == (2, 100.0)
    finally:
        buffer.unlink()


def test_metadata_is_aligned_and_frame_counter_remains_one_per_write() -> None:
    prefix = f"lerobot_test_{uuid4().hex}_"
    buffer = SharedImageBuffer(
        camera_names=["camera"],
        image_shape=(1, 1, 3),
        create=True,
        buffer_name_prefix=prefix,
    )
    reader = None
    try:
        # image_size=3 forces padding and proves alignment is deliberate.
        assert buffer.image_size == 3
        assert buffer.metadata_offset == 8
        assert buffer._sequence_offset % 8 == 0
        assert buffer._timestamp_offset % 8 == 0
        assert buffer._monotonic_offset % 8 == 0
        assert buffer._counter_offset % 8 == 0

        reader = SharedImageBuffer(
            camera_names=["camera"],
            image_shape=(1, 1, 3),
            create=False,
            buffer_name_prefix=prefix,
        )
        first = np.array([[[1, 2, 3]]], dtype=np.uint8)
        second = np.array([[[4, 5, 6]]], dtype=np.uint8)
        buffer.write("camera", first, timestamp=1.0, received_monotonic=11.0)
        assert reader.get_frame_counters() == {"camera": 1}
        buffer.write("camera", second, timestamp=2.0, received_monotonic=12.0)
        assert reader.get_frame_counters() == {"camera": 2}

        raw_version = reader._atomic_load_uint64(
            reader._shm_blocks["camera"].buf,
            reader._sequence_offset,
        )
        assert raw_version == 4  # Internal version advances by two; public counter does not.
        image, timestamp, counter = reader.read("camera")
        np.testing.assert_array_equal(image, second)
        assert timestamp == 2.0
        assert counter == 2
    finally:
        if reader is not None:
            reader.close()
        buffer.unlink()


def test_reader_retries_when_a_write_completes_during_copy(monkeypatch) -> None:
    prefix = f"lerobot_test_{uuid4().hex}_"
    buffer = SharedImageBuffer(
        camera_names=["camera"],
        image_shape=(4, 5, 3),
        create=True,
        buffer_name_prefix=prefix,
    )
    try:
        first = np.full(buffer.image_shape, 10, dtype=np.uint8)
        second = np.full(buffer.image_shape, 20, dtype=np.uint8)
        buffer.write("camera", first, timestamp=1.0, received_monotonic=101.0)

        real_atomic_load = buffer._atomic_load_uint64
        load_count = 0

        def complete_second_write_during_read(buf, offset):
            nonlocal load_count
            load_count += 1
            if load_count == 2:
                sequence = real_atomic_load(buf, offset)
                buffer._atomic_store_uint64(buf, offset, sequence + 1)
                buf[: buffer.image_size] = second.reshape(-1).tobytes()
                struct.pack_into("=d", buf, buffer._timestamp_offset, 2.0)
                struct.pack_into("=d", buf, buffer._monotonic_offset, 102.0)
                struct.pack_into("=Q", buf, buffer._counter_offset, 2)
                buffer._atomic_store_uint64(buf, offset, sequence + 2)
            return real_atomic_load(buf, offset)

        monkeypatch.setattr(buffer, "_atomic_load_uint64", complete_second_write_during_read)
        image, timestamp, counter = buffer.read("camera")

        assert load_count >= 4  # First snapshot was rejected and a second one was read.
        np.testing.assert_array_equal(image, second)
        assert timestamp == 2.0
        assert counter == 2
    finally:
        buffer.unlink()


def test_persistent_odd_version_after_writer_exit_fails_closed(monkeypatch) -> None:
    prefix = f"lerobot_test_{uuid4().hex}_"
    image_shape = (32, 32, 3)
    buffer = SharedImageBuffer(
        camera_names=["camera"],
        image_shape=image_shape,
        create=True,
        buffer_name_prefix=prefix,
    )
    process = None
    try:
        image = np.full(image_shape, 7, dtype=np.uint8)
        buffer.write("camera", image, timestamp=1.0, received_monotonic=2.0)

        ctx = mp.get_context("spawn")
        marked_odd = ctx.Event()
        process = ctx.Process(
            target=_leave_write_in_progress,
            args=(prefix, image_shape, marked_odd),
        )
        process.start()
        assert marked_odd.wait(timeout=5.0)
        process.join(timeout=5.0)
        assert process.exitcode == 0

        monkeypatch.setattr(buffer, "MAX_SNAPSHOT_RETRIES", 3)
        monkeypatch.setattr(buffer, "SNAPSHOT_RETRY_DELAY_SECONDS", 0.0)
        readers = (
            lambda: buffer.read("camera"),
            lambda: buffer.has_new_frame("camera"),
            buffer.get_frame_metadata,
            buffer.get_frame_counters,
        )
        for read_operation in readers:
            with pytest.raises(RuntimeError, match="coherent shared-memory snapshot"):
                read_operation()
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        buffer.unlink()


def test_concurrent_reader_never_observes_mixed_frame_fields() -> None:
    prefix = f"lerobot_test_{uuid4().hex}_"
    image_shape = (256, 256, 3)
    buffer = SharedImageBuffer(
        camera_names=["camera"],
        image_shape=image_shape,
        create=True,
        buffer_name_prefix=prefix,
    )
    process = None
    try:
        ctx = mp.get_context("spawn")
        started = ctx.Event()
        finished = ctx.Event()
        process = ctx.Process(
            target=_write_correlated_frames,
            args=(prefix, image_shape, 300, started, finished),
        )
        process.start()
        assert started.wait(timeout=5.0)

        coherent_reads = 0
        deadline = time.monotonic() + 10.0
        while not finished.is_set() and time.monotonic() < deadline:
            image, timestamp, counter = buffer.read("camera")
            if counter > 0:
                assert timestamp == counter + 0.25
                assert np.all(image == counter % 251)
                coherent_reads += 1

            metadata_counter, received_monotonic = buffer.get_frame_metadata()["camera"]
            if metadata_counter > 0:
                assert received_monotonic == metadata_counter + 1000.5

        process.join(timeout=5.0)
        assert process.exitcode == 0
        assert finished.is_set()
        assert coherent_reads >= 10

        image, timestamp, counter = buffer.read("camera")
        assert counter == 300
        assert timestamp == 300.25
        assert np.all(image == 300 % 251)
        assert buffer.get_frame_counters() == {"camera": 300}
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        buffer.unlink()
