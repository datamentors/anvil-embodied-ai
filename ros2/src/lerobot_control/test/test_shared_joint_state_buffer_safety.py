"""Coherence and fail-closed tests for the serialized joint-state slot."""

import multiprocessing as mp
import struct
import time
from multiprocessing import shared_memory
from uuid import uuid4

import pytest
from lerobot_control.shared_image_buffer import SharedJointStateBuffer


def _correlated_payload(counter: int) -> bytes:
    marker = counter % 251
    return struct.pack("=Q", counter) + bytes([marker]) * (1024 + counter % 127)


def _write_correlated_joint_messages(
    buffer_name: str,
    payload_capacity: int,
    message_count: int,
    started,
    finished,
) -> None:
    slot = SharedJointStateBuffer(
        create=False,
        buffer_name=buffer_name,
        payload_capacity=payload_capacity,
    )
    try:
        started.set()
        for counter in range(1, message_count + 1):
            written_counter = slot.write(
                _correlated_payload(counter),
                received_monotonic=counter + 1000.5,
            )
            if written_counter != counter:
                raise AssertionError((written_counter, counter))
            time.sleep(0.0005)
    finally:
        slot.close()
        finished.set()


def _leave_joint_write_in_progress(
    buffer_name: str,
    payload_capacity: int,
    marked_odd,
) -> None:
    slot = SharedJointStateBuffer(
        create=False,
        buffer_name=buffer_name,
        payload_capacity=payload_capacity,
    )
    try:
        buf = slot._buffer()
        sequence = slot._atomic_load_uint64(buf, slot._sequence_offset)
        slot._atomic_store_uint64(buf, slot._sequence_offset, sequence + 1)
        buf[:16] = b"partial-message!"
        struct.pack_into("=Q", buf, slot._length_offset, 16)
        marked_odd.set()
    finally:
        slot.close()


def test_exact_payload_metadata_sequence_and_read_if_new_are_preserved() -> None:
    buffer_name = f"lerobot_joint_test_{uuid4().hex}"
    slot = SharedJointStateBuffer(
        create=True,
        buffer_name=buffer_name,
        payload_capacity=257,
    )
    reader = None
    try:
        assert slot.metadata_offset == 264
        assert slot._sequence_offset % 8 == 0
        assert slot._length_offset % 8 == 0
        assert slot._monotonic_offset % 8 == 0
        assert slot._counter_offset % 8 == 0
        assert slot.read() == (b"", None, 0)
        assert slot.get_metadata() == (0, None)
        assert slot.read_if_new() is None

        reader = SharedJointStateBuffer(
            create=False,
            buffer_name=buffer_name,
            payload_capacity=257,
        )
        malformed_message = b"\x00cdr\x00names=2;positions=1\xff"
        assert slot.write(malformed_message, received_monotonic=12.25) == 1
        assert reader.read() == (malformed_message, 12.25, 1)
        assert reader.get_metadata() == (1, 12.25)
        assert reader.read_if_new() == (malformed_message, 12.25, 1)
        assert reader.read_if_new() is None

        replacement = b"second-complete-message"
        assert slot.write(replacement, received_monotonic=13.5) == 2
        assert reader.read_if_new() == (replacement, 13.5, 2)
        assert reader._atomic_load_uint64(reader._buffer(), reader._sequence_offset) == 4
    finally:
        if reader is not None:
            reader.close()
        slot.unlink()


@pytest.mark.parametrize(
    ("payload", "received_monotonic", "error_type", "match"),
    [
        (b"", 1.0, ValueError, "must not be empty"),
        (b"12345", 1.0, ValueError, "slot capacity is 4 bytes"),
        (b"ok", float("nan"), ValueError, "must be finite"),
        (b"ok", float("inf"), ValueError, "must be finite"),
        (3, 1.0, TypeError, "must be bytes-like"),
    ],
)
def test_invalid_write_is_rejected_before_publishing(
    payload,
    received_monotonic,
    error_type,
    match,
) -> None:
    buffer_name = f"lerobot_joint_test_{uuid4().hex}"
    slot = SharedJointStateBuffer(
        create=True,
        buffer_name=buffer_name,
        payload_capacity=4,
    )
    try:
        with pytest.raises(error_type, match=match):
            slot.write(payload, received_monotonic)
        assert slot.read() == (b"", None, 0)
        assert slot._atomic_load_uint64(slot._buffer(), slot._sequence_offset) == 0
    finally:
        slot.unlink()


def test_attach_rejects_a_different_payload_capacity() -> None:
    buffer_name = f"lerobot_joint_test_{uuid4().hex}"
    slot = SharedJointStateBuffer(
        create=True,
        buffer_name=buffer_name,
        payload_capacity=64,
    )
    try:
        with pytest.raises(RuntimeError, match="incompatible size"):
            SharedJointStateBuffer(
                create=False,
                buffer_name=buffer_name,
                payload_capacity=128,
            )
    finally:
        slot.unlink()


def test_stable_corrupt_metadata_fails_closed() -> None:
    buffer_name = f"lerobot_joint_test_{uuid4().hex}"
    slot = SharedJointStateBuffer(
        create=True,
        buffer_name=buffer_name,
        payload_capacity=64,
    )
    try:
        slot.write(b"valid", received_monotonic=10.0)
        buf = slot._buffer()

        struct.pack_into("=d", buf, slot._monotonic_offset, float("nan"))
        with pytest.raises(RuntimeError, match="non-finite receive timestamp"):
            slot.get_metadata()

        struct.pack_into("=d", buf, slot._monotonic_offset, 10.0)
        struct.pack_into("=Q", buf, slot._length_offset, slot.payload_capacity + 1)
        with pytest.raises(RuntimeError, match="payload length 65 exceeds"):
            slot.read()
    finally:
        slot.unlink()


def test_persistent_odd_version_after_writer_exit_fails_closed(monkeypatch) -> None:
    buffer_name = f"lerobot_joint_test_{uuid4().hex}"
    payload_capacity = 2048
    slot = SharedJointStateBuffer(
        create=True,
        buffer_name=buffer_name,
        payload_capacity=payload_capacity,
    )
    process = None
    try:
        slot.write(b"complete-before-crash", received_monotonic=2.0)
        ctx = mp.get_context("spawn")
        marked_odd = ctx.Event()
        process = ctx.Process(
            target=_leave_joint_write_in_progress,
            args=(buffer_name, payload_capacity, marked_odd),
        )
        process.start()
        assert marked_odd.wait(timeout=5.0)
        process.join(timeout=5.0)
        assert process.exitcode == 0

        monkeypatch.setattr(slot, "MAX_SNAPSHOT_RETRIES", 3)
        monkeypatch.setattr(slot, "SNAPSHOT_RETRY_DELAY_SECONDS", 0.0)
        for read_operation in (slot.read, slot.read_if_new, slot.get_metadata):
            with pytest.raises(RuntimeError, match="coherent shared-memory snapshot"):
                read_operation()
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        slot.unlink()


def test_concurrent_reader_never_observes_mixed_joint_message_fields() -> None:
    buffer_name = f"lerobot_joint_test_{uuid4().hex}"
    payload_capacity = 4096
    slot = SharedJointStateBuffer(
        create=True,
        buffer_name=buffer_name,
        payload_capacity=payload_capacity,
    )
    process = None
    try:
        ctx = mp.get_context("spawn")
        started = ctx.Event()
        finished = ctx.Event()
        process = ctx.Process(
            target=_write_correlated_joint_messages,
            args=(buffer_name, payload_capacity, 300, started, finished),
        )
        process.start()
        assert started.wait(timeout=5.0)

        coherent_reads = 0
        deadline = time.monotonic() + 10.0
        while not finished.is_set() and time.monotonic() < deadline:
            payload, received_monotonic, counter = slot.read()
            if counter > 0:
                encoded_counter = struct.unpack_from("=Q", payload)[0]
                assert encoded_counter == counter
                assert received_monotonic == counter + 1000.5
                assert payload[8:] == bytes([counter % 251]) * (1024 + counter % 127)
                coherent_reads += 1

        process.join(timeout=5.0)
        assert process.exitcode == 0
        assert finished.is_set()
        assert coherent_reads >= 10

        payload, received_monotonic, counter = slot.read()
        assert counter == 300
        assert struct.unpack_from("=Q", payload)[0] == 300
        assert received_monotonic == 1300.5
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        slot.unlink()


def test_unlink_removes_the_owned_slot() -> None:
    buffer_name = f"lerobot_joint_test_{uuid4().hex}"
    slot = SharedJointStateBuffer(
        create=True,
        buffer_name=buffer_name,
        payload_capacity=32,
    )
    slot.unlink()

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=buffer_name)

