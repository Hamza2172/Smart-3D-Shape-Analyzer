import cv2
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def make_kernel(kernel_size: int, shape: int = cv2.MORPH_RECT) -> np.ndarray:
    kernel_size = max(1, kernel_size)
    return cv2.getStructuringElement(shape, (kernel_size, kernel_size))


def _validate_binary(image: Optional[np.ndarray], caller: str) -> bool:
    if image is None or image.ndim != 2:
        logger.error(
            "%s: expected 2D binary array, got %s",
            caller,
            getattr(image, "shape", None),
        )
        return False
    return True


def apply_dilation(
    binary: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> Optional[np.ndarray]:
    if not _validate_binary(binary, "apply_dilation"):
        return None
    try:
        kernel = make_kernel(kernel_size)
        result = cv2.dilate(binary, kernel, iterations=iterations)
        logger.info("Dilation OK — kernel=%d iter=%d", kernel_size, iterations)
        return result
    except Exception as exc:
        logger.exception("apply_dilation failed: %s", exc)
        return None


def apply_erosion(
    binary: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> Optional[np.ndarray]:
    if not _validate_binary(binary, "apply_erosion"):
        return None
    try:
        kernel = make_kernel(kernel_size)
        result = cv2.erode(binary, kernel, iterations=iterations)
        logger.info("Erosion OK — kernel=%d iter=%d", kernel_size, iterations)
        return result
    except Exception as exc:
        logger.exception("apply_erosion failed: %s", exc)
        return None


def apply_opening(
    binary: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> Optional[np.ndarray]:
    if not _validate_binary(binary, "apply_opening"):
        return None
    try:
        kernel = make_kernel(kernel_size)
        result = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel, iterations=iterations
        )
        logger.info("Opening OK — kernel=%d iter=%d", kernel_size, iterations)
        return result
    except Exception as exc:
        logger.exception("apply_opening failed: %s", exc)
        return None


def apply_closing(
    binary: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> Optional[np.ndarray]:
    if not _validate_binary(binary, "apply_closing"):
        return None
    try:
        kernel = make_kernel(kernel_size)
        result = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel, iterations=iterations
        )
        logger.info("Closing OK — kernel=%d iter=%d", kernel_size, iterations)
        return result
    except Exception as exc:
        logger.exception("apply_closing failed: %s", exc)
        return None


OPERATIONS: dict[str, callable] = {
    "Dilation": apply_dilation,
    "Erosion":  apply_erosion,
    "Opening":  apply_opening,
    "Closing":  apply_closing,
}

OPERATION_DESCRIPTIONS: dict[str, str] = {
    "Dilation": (
        "Expands white (foreground) regions outward. "
        "Bridges small gaps between shapes and thickens thin edges."
    ),
    "Erosion": (
        "Shrinks white (foreground) regions inward. "
        "Separates touching objects and removes small noise specks."
    ),
    "Opening": (
        "Erosion then Dilation. Removes small isolated noise blobs "
        "while keeping the overall shape sizes largely intact."
    ),
    "Closing": (
        "Dilation then Erosion. Fills small holes and narrow gaps "
        "inside foreground regions without changing the outer boundary."
    ),
}


def run_morphology(
    binary: np.ndarray,
    operation: str,
    kernel_size: int = 3,
    iterations: int = 1,
) -> dict:
    fn = OPERATIONS.get(operation)
    if fn is None:
        logger.error("run_morphology: unknown operation '%s'", operation)
        return {
            "operation": operation, "kernel_size": kernel_size,
            "iterations": iterations, "binary_in": binary,
            "result": None, "diff": None, "pixels_changed": 0,
        }

    result = fn(binary, kernel_size=kernel_size, iterations=iterations)

    diff = None
    pixels_changed = 0
    if result is not None and binary is not None:
        diff = cv2.absdiff(binary, result)
        pixels_changed = int(np.count_nonzero(diff))

    return {
        "operation":      operation,
        "kernel_size":    kernel_size,
        "iterations":     iterations,
        "binary_in":      binary,
        "result":         result,
        "diff":           diff,
        "pixels_changed": pixels_changed,
    }