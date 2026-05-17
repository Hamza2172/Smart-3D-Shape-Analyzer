import cv2
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from skimage.morphology import skeletonize, thin
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False
    logger.warning(
        "scikit-image not installed — Stage 6 will be disabled. "
        "Run: pip install scikit-image"
    )


def _to_bool(binary: np.ndarray) -> Optional[np.ndarray]:
    if binary is None or binary.ndim != 2:
        return None
    return binary > 0


def _to_uint8(bool_arr: np.ndarray) -> np.ndarray:
    return (bool_arr.astype(np.uint8)) * 255


def create_skeleton(binary: np.ndarray) -> Optional[np.ndarray]:
    if not _SKIMAGE_AVAILABLE:
        logger.error("create_skeleton: scikit-image not available.")
        return None

    bool_img = _to_bool(binary)
    if bool_img is None:
        logger.error("create_skeleton: invalid input array.")
        return None

    try:
        skel = skeletonize(bool_img)
        result = _to_uint8(skel)
        px_count = int(np.count_nonzero(result))
        logger.info("create_skeleton: %d skeleton pixels", px_count)
        return result
    except Exception as exc:
        logger.exception("create_skeleton failed: %s", exc)
        return None


def create_thinning(binary: np.ndarray) -> Optional[np.ndarray]:
    if not _SKIMAGE_AVAILABLE:
        logger.error("create_thinning: scikit-image not available.")
        return None

    bool_img = _to_bool(binary)
    if bool_img is None:
        logger.error("create_thinning: invalid input array.")
        return None

    try:
        thinned = thin(bool_img)
        result  = _to_uint8(thinned)
        px_count = int(np.count_nonzero(result))
        logger.info("create_thinning: %d thinned pixels", px_count)
        return result
    except Exception as exc:
        logger.exception("create_thinning failed: %s", exc)
        return None


def draw_skeleton_overlay(
    rgb_image: np.ndarray,
    skeleton: np.ndarray,
    colour_rgb: tuple[int, int, int] = (0, 230, 255),
    thickness: int = 1,
) -> Optional[np.ndarray]:
    if rgb_image is None or skeleton is None:
        logger.error("draw_skeleton_overlay: None input.")
        return None

    if rgb_image.shape[:2] != skeleton.shape[:2]:
        logger.error(
            "draw_skeleton_overlay: shape mismatch rgb=%s skel=%s",
            rgb_image.shape, skeleton.shape,
        )
        return None

    try:
        bg = (rgb_image * 0.35).astype(np.uint8)

        display_skel = skeleton.copy()
        if thickness > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (thickness, thickness)
            )
            display_skel = cv2.dilate(display_skel, kernel, iterations=1)

        mask    = display_skel > 0
        overlay = bg.copy()
        overlay[mask] = colour_rgb

        logger.info("draw_skeleton_overlay: overlay created, skel_px=%d", np.sum(mask))
        return overlay

    except Exception as exc:
        logger.exception("draw_skeleton_overlay failed: %s", exc)
        return None


def run_skeleton_pipeline(
    rgb_image: np.ndarray,
    binary: np.ndarray,
    mode: str = "skeleton",
    overlay_thickness: int = 1,
) -> dict:
    out: dict = {
        "mode":          mode,
        "result":        None,
        "overlay":       None,
        "available":     _SKIMAGE_AVAILABLE,
        "skel_pixels":   0,
        "binary_pixels": int(np.count_nonzero(binary)) if binary is not None else 0,
        "reduction_pct": 0.0,
    }

    if not _SKIMAGE_AVAILABLE or binary is None:
        return out

    result = create_skeleton(binary) if mode == "skeleton" else create_thinning(binary)
    out["result"] = result

    if result is not None:
        skel_px = int(np.count_nonzero(result))
        out["skel_pixels"] = skel_px
        if out["binary_pixels"] > 0:
            out["reduction_pct"] = round(
                (1 - skel_px / out["binary_pixels"]) * 100, 1
            )

        out["overlay"] = draw_skeleton_overlay(
            rgb_image, result, thickness=overlay_thickness
        )

    return out