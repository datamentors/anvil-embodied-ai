"""Safety and JPEG-integrity tests for the multi-process image worker."""

import ast
import inspect
import os
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from lerobot_control import image_worker, jpeg_integrity


def test_opencv_thread_limit_is_scoped_to_spawned_image_worker() -> None:
    """Importing the module must not change the main inference process."""
    module_tree = ast.parse(inspect.getsource(image_worker))
    top_level_calls = [
        node
        for statement in module_tree.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cv2"
        and node.func.attr == "setNumThreads"
        and not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert top_level_calls == []

    worker_source = inspect.getsource(image_worker.run_image_worker)
    assert worker_source.index("cv2.setNumThreads(1)") < worker_source.index("rclpy.init")


def _make_node() -> tuple[image_worker.ImageWorkerNode, Mock]:
    node = image_worker.ImageWorkerNode.__new__(image_worker.ImageWorkerNode)
    node.camera_name = "base"
    node.image_shape = (4, 4, 3)
    node._debug_dir = None
    node._debug_max_frames = 0
    node._debug_saved = 0
    node._debug_last_save = 0.0
    node.shared_buffer = Mock()
    node.frame_count = 0
    node.bad_frame_count = 0
    node.decode_error_count = 0
    node.jpeg_warning_count = 0
    node.invalid_jpeg_count = 0
    node._jpeg_decoder = jpeg_integrity.StrictJpegDecoder()
    node._last_bad_frame_log_monotonic = None
    node._bad_frames_since_log = 0
    node._bad_reasons_since_log = Counter()
    node._last_bad_detail = ""
    logger = Mock()
    node.get_logger = Mock(return_value=logger)
    return node, logger


def _message(payload: bytes):
    return SimpleNamespace(
        data=payload,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=42, nanosec=500_000_000)),
    )


def _make_joint_worker_node() -> tuple[image_worker.JointStateWorkerNode, Mock]:
    node = image_worker.JointStateWorkerNode.__new__(image_worker.JointStateWorkerNode)
    node.shared_buffer = Mock()
    node.frame_count = 0
    logger = Mock()
    node.get_logger = Mock(return_value=logger)
    return node, logger


def test_valid_jpeg_payload_is_written_to_shared_memory() -> None:
    node, logger = _make_node()
    try:
        source = np.zeros((4, 4, 3), dtype=np.uint8)
        encoded_ok, encoded = cv2.imencode(".jpg", source)
        assert encoded_ok

        node._image_callback(_message(encoded.tobytes()))

        node.shared_buffer.write.assert_called_once()
        assert node.frame_count == 1
        assert node.bad_frame_count == 0
        assert node.decode_error_count == 0
        logger.warning.assert_not_called()
    finally:
        node._jpeg_decoder.close()


def test_image_receipt_time_is_captured_before_decode(monkeypatch) -> None:
    """JPEG work must not inflate the watchdog's inter-sensor receive skew."""
    node, _logger = _make_node()
    monotonic = iter((10.0, 10.125))
    monkeypatch.setattr(image_worker.time, "monotonic", lambda: next(monotonic))
    decoded = np.zeros((4, 4, 3), dtype=np.uint8)

    def decode(*_args):
        assert image_worker.time.monotonic() == 10.125
        return decoded

    monkeypatch.setattr(jpeg_integrity.cv2, "imdecode", decode)
    monkeypatch.setattr(image_worker.cv2, "cvtColor", lambda value, _code: value)

    try:
        node._image_callback(_message(b"\xff\xd8jpeg\xff\xd9"))

        node.shared_buffer.write.assert_called_once()
        args = node.shared_buffer.write.call_args
        assert args.args[0] == "base"
        assert args.args[1] is decoded
        assert args.args[2] == 42.5
        assert args.kwargs == {"received_monotonic": 10.0}
        assert node.frame_count == 1
    finally:
        node._jpeg_decoder.close()


def test_native_corrupt_warning_rejects_frame_and_preserves_all_warning_lines(
    monkeypatch,
) -> None:
    node, logger = _make_node()
    decoded = np.zeros((4, 4, 3), dtype=np.uint8)

    def decode(*_args):
        os.write(2, b"libjpeg diagnostic prefix\n")
        os.write(2, b"Corrupt JPEG data: simulated premature end\n")
        return decoded

    monkeypatch.setattr(jpeg_integrity.cv2, "imdecode", decode)
    try:
        node._image_callback(_message(b"\xff\xd8jpeg\xff\xd9"))

        node.shared_buffer.write.assert_not_called()
        assert node.frame_count == 0
        assert node.bad_frame_count == 1
        assert node.decode_error_count == 0
        assert node.jpeg_warning_count == 1
        warning = logger.warning.call_args.args[0]
        assert "libjpeg_corrupt_warning=1" in warning
        assert "libjpeg diagnostic prefix" in warning
        assert "Corrupt JPEG data: simulated premature end" in warning
    finally:
        node._jpeg_decoder.close()


def test_any_native_stderr_rejects_frame_even_without_known_corrupt_text(monkeypatch) -> None:
    node, logger = _make_node()
    decoded = np.zeros((4, 4, 3), dtype=np.uint8)

    def decode(*_args):
        os.write(2, b"previously unknown native JPEG diagnostic\n")
        return decoded

    monkeypatch.setattr(jpeg_integrity.cv2, "imdecode", decode)
    try:
        node._image_callback(_message(b"\xff\xd8jpeg\xff\xd9"))

        node.shared_buffer.write.assert_not_called()
        assert node.frame_count == 0
        assert node.bad_frame_count == 1
        assert node.decode_error_count == 0
        assert node.jpeg_warning_count == 1
        warning = logger.warning.call_args.args[0]
        assert "libjpeg_native_warning=1" in warning
        assert "previously unknown native JPEG diagnostic" in warning
    finally:
        node._jpeg_decoder.close()


@pytest.mark.parametrize(
    ("payload", "expected_reason"),
    [
        (b"not-a-jpeg\xff\xd9", "missing_jpeg_soi"),
        (b"\xff\xd8not-a-complete-jpeg", "missing_jpeg_eoi"),
    ],
)
def test_invalid_soi_or_eoi_is_rejected_before_decode(
    monkeypatch, payload: bytes, expected_reason: str
) -> None:
    node, logger = _make_node()
    decode = Mock()
    monkeypatch.setattr(jpeg_integrity.cv2, "imdecode", decode)
    try:
        node._image_callback(_message(payload))

        decode.assert_not_called()
        node.shared_buffer.write.assert_not_called()
        assert node.frame_count == 0
        assert node.bad_frame_count == 1
        assert node.decode_error_count == 0
        assert node.invalid_jpeg_count == 1
        assert f"{expected_reason}=1" in logger.warning.call_args.args[0]
    finally:
        node._jpeg_decoder.close()


def test_decode_none_is_counted_and_never_written(monkeypatch) -> None:
    node, logger = _make_node()
    monkeypatch.setattr(jpeg_integrity.cv2, "imdecode", lambda *_args: None)
    try:
        node._image_callback(_message(b"\xff\xd8jpeg\xff\xd9"))

        node.shared_buffer.write.assert_not_called()
        assert node.frame_count == 0
        assert node.bad_frame_count == 1
        assert node.decode_error_count == 1
        assert "decode_failed=1" in logger.warning.call_args.args[0]
    finally:
        node._jpeg_decoder.close()


def test_decode_exception_restores_stderr_and_never_writes(monkeypatch, capfd) -> None:
    node, logger = _make_node()

    def decode(*_args):
        os.write(2, b"native detail before exception\n")
        raise RuntimeError("simulated decoder failure")

    monkeypatch.setattr(jpeg_integrity.cv2, "imdecode", decode)
    try:
        node._image_callback(_message(b"\xff\xd8jpeg\xff\xd9"))
        os.write(2, b"stderr-restored-sentinel\n")
        _stdout, stderr = capfd.readouterr()

        node.shared_buffer.write.assert_not_called()
        assert node.frame_count == 0
        assert node.bad_frame_count == 1
        assert node.decode_error_count == 1
        assert "decode_exception=1" in logger.warning.call_args.args[0]
        assert "native detail before exception" in logger.warning.call_args.args[0]
        assert "simulated decoder failure" in logger.warning.call_args.args[0]
        assert "stderr-restored-sentinel" in stderr
        assert "native detail before exception" not in stderr
    finally:
        node._jpeg_decoder.close()


def test_bad_frame_logging_is_rate_limited_without_losing_counts(monkeypatch) -> None:
    node, logger = _make_node()
    decoded = np.zeros((4, 4, 3), dtype=np.uint8)

    def decode(*_args):
        os.write(2, b"Corrupt JPEG data: simulated\n")
        return decoded

    monotonic = iter((0.0, 0.0, 0.1, 0.1, 6.0, 6.0))
    monkeypatch.setattr(image_worker.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(jpeg_integrity.cv2, "imdecode", decode)
    try:
        for _ in range(3):
            node._image_callback(_message(b"\xff\xd8jpeg\xff\xd9"))

        assert logger.warning.call_count == 2
        assert "since_last_log=2" in logger.warning.call_args.args[0]
        assert "bad_total=3" in logger.warning.call_args.args[0]
        assert "libjpeg_warnings=3" in logger.warning.call_args.args[0]
        assert node.bad_frame_count == 3
        assert node.frame_count == 0
        node.shared_buffer.write.assert_not_called()
    finally:
        node._jpeg_decoder.close()


def test_joint_worker_qos_matches_main_process_contract() -> None:
    profile = image_worker.joint_state_qos_profile()

    assert profile.reliability == image_worker.ReliabilityPolicy.RELIABLE
    assert profile.history == image_worker.HistoryPolicy.KEEP_LAST
    assert profile.depth == 10


def test_joint_worker_preserves_complete_serialized_message_and_ingress_time(
    monkeypatch,
) -> None:
    node, logger = _make_joint_worker_node()
    message = SimpleNamespace(name=["duplicate", "duplicate"], position=[1.0])
    serialized = b"cdr-with-invalid-name-position-lengths"
    monotonic = iter((10.0, 20.0))
    monkeypatch.setattr(image_worker.time, "monotonic", lambda: next(monotonic))

    def serialize(value):
        assert value is message
        assert image_worker.time.monotonic() == 20.0
        return serialized

    monkeypatch.setattr(image_worker, "serialize_message", serialize)

    node._joint_callback(message)

    node.shared_buffer.write.assert_called_once_with(serialized, 10.0)
    assert node.frame_count == 1
    logger.error.assert_not_called()


def test_joint_worker_serialization_failure_never_publishes_partial_slot(monkeypatch) -> None:
    node, logger = _make_joint_worker_node()
    monkeypatch.setattr(image_worker.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(
        image_worker,
        "serialize_message",
        Mock(side_effect=RuntimeError("simulated CDR failure")),
    )

    node._joint_callback(SimpleNamespace())

    node.shared_buffer.write.assert_not_called()
    assert node.frame_count == 0
    assert "simulated CDR failure" in logger.error.call_args.args[0]
