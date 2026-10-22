"""Image processing utilities"""

import io

import cv2
import numpy as np
from PIL import Image


def decode_image(img_record: bytes, encoding: str, height: int, width: int) -> np.ndarray:
    """
    Decode ROS Image message data

    Args:
        img_record: Raw image bytes from ROS message
        encoding: Image encoding (e.g., 'bgr8', 'rgb8', 'mono8')
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        Decoded image as numpy array (H, W, C) for RGB or (H, W) for mono

    Raises:
        ValueError: If encoding is not supported
    """
    img_data = np.frombuffer(img_record, dtype=np.uint8)

    if encoding == "bgr8":
        img_bgr = img_data.reshape((height, width, 3))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    elif encoding == "rgb8":
        return img_data.reshape((height, width, 3))

    elif encoding == "mono8" or encoding == "mono16":
        return img_data.reshape((height, width))

    elif encoding in ["jpeg", "jpg"]:
        # JPEG compressed
        return cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    elif encoding in ["png"]:
        # PNG compressed
        img = Image.open(io.BytesIO(img_record))
        return np.array(img)

    elif encoding in ["yuv422_yuy2", "yuyv"]:
        # YUV 4:2:2 (YUYV) format - 2 bytes per pixel
        img_yuyv = img_data.reshape((height, width, 2))
        img_rgb = cv2.cvtColor(img_yuyv, cv2.COLOR_YUV2RGB_YUYV)
        return img_rgb

    elif encoding in ["uyvy"]:
        # YUV 4:2:2 (UYVY) format - 2 bytes per pixel
        img_uyvy = img_data.reshape((height, width, 2))
        img_rgb = cv2.cvtColor(img_uyvy, cv2.COLOR_YUV2RGB_UYVY)
        return img_rgb

    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")


def decode_compressed_image(data: bytes, format: str) -> np.ndarray:
    """
    Decode CompressedImage message data

    Args:
        data: Raw compressed image bytes
        format: Compression format (e.g., 'jpeg', 'png', or ROS format like 'rgb8; jpeg compressed bgr8')

    Returns:
        Decoded image as numpy array (H, W, C) in RGB format

    Raises:
        ValueError: If format is not supported
    """
    # Parse ROS CompressedImage format string
    # Format can be: 'jpeg', 'png', or 'rgb8; jpeg compressed bgr8'
    format_lower = format.lower()

    # Extract compression type from ROS format string
    if "jpeg" in format_lower or "jpg" in format_lower:
        compression = "jpeg"
    elif "png" in format_lower:
        compression = "png"
    else:
        compression = format_lower

    img_data = np.frombuffer(data, dtype=np.uint8)

    if compression in ["jpeg", "jpg"]:
        img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            # cv2 failed — try PIL as a more lenient fallback
            try:
                img_pil = Image.open(io.BytesIO(bytes(data)))
                img_pil.load()  # force full decode so errors surface here
                return np.array(img_pil.convert("RGB"))
            except Exception as pil_err:
                raise ValueError(
                    f"Failed to decode JPEG image (cv2 and PIL both failed: {pil_err})"
                ) from pil_err
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    elif compression == "png":
        img = Image.open(io.BytesIO(data))
        return np.array(img)

    else:
        raise ValueError(f"Unsupported compressed image format: {format}")


def encode_image_to_bytes(img: np.ndarray, format: str = "png") -> bytes:
    """
    Encode numpy image array to bytes

    Args:
        img: Image array (H, W, C) or (H, W)
        format: Output format ('png', 'jpg', 'jpeg')

    Returns:
        Encoded image bytes
    """
    if format.lower() in ["jpg", "jpeg"]:
        success, buffer = cv2.imencode(".jpg", img)
    elif format.lower() == "png":
        success, buffer = cv2.imencode(".png", img)
    else:
        raise ValueError(f"Unsupported format: {format}")

    if not success:
        raise RuntimeError("Failed to encode image")

    return buffer.tobytes()


# Relative tolerance for treating two aspect ratios as equal. Covers rounding the
# derived height to an even number (e.g. 848×480 → 640×362).
ASPECT_RATIO_TOLERANCE = 0.01


def aspect_ratio_matches(size_a: tuple, size_b: tuple) -> bool:
    """Whether two (width, height) sizes share an aspect ratio within ASPECT_RATIO_TOLERANCE."""
    ratio_a = size_a[0] / size_a[1]
    ratio_b = size_b[0] / size_b[1]
    return abs(ratio_a - ratio_b) <= ASPECT_RATIO_TOLERANCE * ratio_a


def output_resolution(source_size: tuple, target_width: int) -> tuple:
    """
    Dataset resolution for a camera: the target width, height from the camera's aspect ratio.

    Args:
        source_size: Native camera (width, height)
        target_width: Output width in pixels

    Returns:
        (width, height) with the source aspect ratio. Height is rounded to an even
        number, as required by yuv420p video encoding.
    """
    src_w, src_h = source_size
    target_h = max(2, round(target_width * src_h / src_w / 2) * 2)
    return (target_width, target_h)


def resize_image(img: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize image to target size. Never pads, crops or stretches.

    Args:
        img: Image array (H, W, C) or (H, W)
        target_size: (width, height) tuple — must have the image's aspect ratio
            (see output_resolution)

    Returns:
        Resized image, shape (H, W, C) or (H, W).

    Raises:
        RuntimeError: If the image aspect ratio differs from target_size. Not a
            ValueError: the converter skips episodes on ValueError (corrupt frames),
            whereas a camera/resolution mismatch must stop the run.
    """
    target_w, target_h = target_size
    src_h, src_w = img.shape[:2]

    if src_w == target_w and src_h == target_h:
        return img

    if not aspect_ratio_matches((src_w, src_h), (target_w, target_h)):
        raise RuntimeError(
            f"Image is {src_w}x{src_h} but the dataset resolution is {target_w}x{target_h} "
            "(different aspect ratio). Images are never padded or stretched — all cameras "
            "must share one aspect ratio."
        )

    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
