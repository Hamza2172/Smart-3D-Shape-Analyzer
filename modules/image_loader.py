import cv2
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def load_image_from_bytes(file_bytes: bytes) -> Optional[np.ndarray]:
    if not file_bytes:
        logger.warning("load_image_from_bytes received empty byte string.")
        return None

    try:
        np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
        bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if bgr_image is None:
            logger.error("cv2.imdecode returned None — unsupported or corrupt file.")
            return None

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        logger.info(
            "Image loaded successfully: shape=%s dtype=%s",
            rgb_image.shape,
            rgb_image.dtype,
        )
        return rgb_image

    except Exception as exc:
        logger.exception("Unexpected error while decoding image: %s", exc)
        return None


def get_image_metadata(image: np.ndarray) -> dict:
    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1

    return {
        "height": height,
        "width": width,
        "channels": channels,
        "dtype": str(image.dtype),
        "aspect_ratio": round(width / height, 3),
    }