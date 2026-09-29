"""Strict JPEG validation with per-process native stderr capture."""

from __future__ import annotations

import ctypes
import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np

CORRUPT_JPEG_WARNING = "Corrupt JPEG data"
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


@dataclass(frozen=True)
class JpegDecodeResult:
    """Result of validating and decoding exactly one compressed payload."""

    image: np.ndarray | None
    native_stderr: str = ""
    marker_error: str | None = None
    decode_exception: str | None = None

    @property
    def has_corrupt_warning(self) -> bool:
        """Whether libjpeg explicitly reported corrupt input."""
        return CORRUPT_JPEG_WARNING in self.native_stderr

    @property
    def has_native_warning(self) -> bool:
        """Whether native JPEG decoding emitted any stderr diagnostic."""
        return bool(self.native_stderr)

    @property
    def decode_failed(self) -> bool:
        """Whether OpenCV raised or returned no decoded image."""
        return self.marker_error is None and (
            self.decode_exception is not None or self.image is None
        )

    @property
    def rejection_reason(self) -> str | None:
        """Return the fail-closed reason, or ``None`` for an acceptable frame."""
        if self.marker_error is not None:
            return self.marker_error
        if self.decode_exception is not None:
            return "decode_exception"
        if self.image is None:
            return "decode_failed"
        if self.has_corrupt_warning:
            return "libjpeg_corrupt_warning"
        if self.has_native_warning:
            # Fail closed on warning variants that the current libjpeg wording
            # does not explicitly identify as corrupt.
            return "libjpeg_native_warning"
        return None


class StrictJpegDecoder:
    """Decode JPEGs while capturing libjpeg's native writes to stderr.

    Each image worker owns one instance and uses a single-threaded executor, so
    redirecting file descriptor 2 cannot capture another camera worker. The
    temporary file is reused to avoid allocating a file for every frame.
    """

    def __init__(self) -> None:
        # This file deliberately remains open for the decoder's full lifetime.
        self._stderr_capture = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
        self._libc = ctypes.CDLL(None)
        self._libc.fflush.argtypes = [ctypes.c_void_p]
        self._libc.fflush.restype = ctypes.c_int

    @staticmethod
    def marker_error(payload: bytes) -> str | None:
        """Validate the complete JPEG envelope before invoking native code."""
        if len(payload) < len(JPEG_SOI) + len(JPEG_EOI):
            return "invalid_jpeg_markers"
        if not payload.startswith(JPEG_SOI):
            return "missing_jpeg_soi"
        if not payload.endswith(JPEG_EOI):
            return "missing_jpeg_eoi"
        return None

    def decode(self, payload: bytes) -> JpegDecodeResult:
        """Validate and decode a payload without leaking or hiding native stderr."""
        marker_error = self.marker_error(payload)
        if marker_error is not None:
            return JpegDecodeResult(image=None, marker_error=marker_error)

        image: np.ndarray | None = None
        decode_exception: str | None = None
        saved_stderr: int | None = None
        redirected = False

        self._stderr_capture.seek(0)
        self._stderr_capture.truncate(0)
        try:
            saved_stderr = os.dup(2)
            self._libc.fflush(None)
            os.dup2(self._stderr_capture.fileno(), 2)
            redirected = True
            try:
                encoded = np.frombuffer(payload, dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            except Exception as exc:  # noqa: BLE001 - report failure and reject the frame.
                decode_exception = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if redirected:
                    self._libc.fflush(None)
            finally:
                if saved_stderr is not None:
                    try:
                        os.dup2(saved_stderr, 2)
                    finally:
                        os.close(saved_stderr)

        self._stderr_capture.seek(0)
        native_stderr = self._stderr_capture.read().decode("utf-8", errors="replace").strip()
        return JpegDecodeResult(
            image=image,
            native_stderr=native_stderr,
            decode_exception=decode_exception,
        )

    def close(self) -> None:
        """Release the per-worker capture file."""
        self._stderr_capture.close()
