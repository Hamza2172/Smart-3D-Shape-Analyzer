import cv2
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def to_grayscale(image: np.ndarray) -> Optional[np.ndarray]:
    if image is None or image.ndim != 3:
        logger.error("to_grayscale: expected 3-channel RGB array, got %s",
                     getattr(image, "shape", None))
        return None

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        logger.info("Grayscale conversion OK — shape=%s", gray.shape)
        return gray
    except Exception as exc:
        logger.exception("to_grayscale failed: %s", exc)
        return None


def reduce_noise(
    gray: np.ndarray,
    kernel_size: int = 3,
) -> Optional[np.ndarray]:
    if gray is None or gray.ndim != 2:
        logger.error("reduce_noise: expected 2D grayscale array.")
        return None

    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    try:
        denoised = cv2.medianBlur(gray, kernel_size)
        logger.info("Noise reduction OK — kernel=%d", kernel_size)
        return denoised
    except Exception as exc:
        logger.exception("reduce_noise failed: %s", exc)
        return None


def apply_gaussian_blur(
    gray: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> Optional[np.ndarray]:
    if gray is None or gray.ndim != 2:
        logger.error("apply_gaussian_blur: expected 2D grayscale array.")
        return None

    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    try:
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
        logger.info("Gaussian blur OK — kernel=%d sigma=%.2f", kernel_size, sigma)
        return blurred
    except Exception as exc:
        logger.exception("apply_gaussian_blur failed: %s", exc)
        return None


def apply_global_threshold(
    gray: np.ndarray,
    threshold_value: int = 127,
    invert: bool = False,
) -> Optional[np.ndarray]:
    if gray is None or gray.ndim != 2:
        logger.error("apply_global_threshold: expected 2D grayscale array.")
        return None

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    try:
        _, binary = cv2.threshold(gray, threshold_value, 255, thresh_type)
        white_pct = round(np.sum(binary == 255) / binary.size * 100, 1)
        logger.info(
            "Global threshold OK — value=%d invert=%s white=%.1f%%",
            threshold_value, invert, white_pct,
        )
        return binary
    except Exception as exc:
        logger.exception("apply_global_threshold failed: %s", exc)
        return None


def apply_adaptive_threshold(
    gray: np.ndarray,
    block_size: int = 11,
    c_constant: int = 2,
    invert: bool = False,
) -> Optional[np.ndarray]:
    if gray is None or gray.ndim != 2:
        logger.error("apply_adaptive_threshold: expected 2D grayscale array.")
        return None

    block_size = max(3, block_size)
    if block_size % 2 == 0:
        block_size += 1

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    try:
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type,
            block_size,
            c_constant,
        )
        logger.info(
            "Adaptive threshold OK — block=%d C=%d invert=%s",
            block_size, c_constant, invert,
        )
        return binary
    except Exception as exc:
        logger.exception("apply_adaptive_threshold failed: %s", exc)
        return None


def run_preprocessing_pipeline(
    rgb_image: np.ndarray,
    blur_kernel: int = 5,
    threshold_value: int = 127,
    use_adaptive: bool = False,
    adaptive_block_size: int = 11,
    adaptive_c: int = 2,
    invert_threshold: bool = False,
    denoise_kernel: int = 3,
) -> dict:
    results: dict = {
        "grayscale": None,
        "denoised": None,
        "blurred": None,
        "binary": None,
        "method": "adaptive" if use_adaptive else "global",
    }

    gray = to_grayscale(rgb_image)
    results["grayscale"] = gray
    if gray is None:
        return results

    denoised = reduce_noise(gray, kernel_size=denoise_kernel)
    results["denoised"] = denoised
    working = denoised if denoised is not None else gray

    blurred = apply_gaussian_blur(working, kernel_size=blur_kernel)
    results["blurred"] = blurred
    working = blurred if blurred is not None else working

    if use_adaptive:
        binary = apply_adaptive_threshold(
            working,
            block_size=adaptive_block_size,
            c_constant=adaptive_c,
            invert=invert_threshold,
        )
    else:
        binary = apply_global_threshold(
            working,
            threshold_value=threshold_value,
            invert=invert_threshold,
        )

    results["binary"] = binary
    return results