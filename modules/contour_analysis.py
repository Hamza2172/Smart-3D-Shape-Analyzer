import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

_PALETTE_BGR: list[tuple[int, int, int]] = [
    (0,   200, 255),
    (50,  220, 80),
    (255, 80,  80),
    (255, 120, 0),
    (200, 0,   200),
    (0,   180, 180),
    (80,  80,  255),
    (180, 255, 100),
    (100, 100, 255),
    (255, 200, 0),
    (0,   255, 180),
    (150, 50,  255),
]

def _get_colour(idx: int) -> tuple[int, int, int]:
    return _PALETTE_BGR[idx % len(_PALETTE_BGR)]


@dataclass
class ContourInfo:
    index:       int
    colour_bgr:  tuple[int, int, int]
    points:      np.ndarray = field(repr=False)
    area:        float  = 0.0
    perimeter:   float  = 0.0
    cx:          int    = 0
    cy:          int    = 0
    bbox_x:      int = 0
    bbox_y:      int = 0
    bbox_w:      int = 0
    bbox_h:      int = 0
    aspect_ratio:    float = 0.0
    extent:          float = 0.0
    circularity:     float = 0.0

    @property
    def colour_rgb(self) -> tuple[int, int, int]:
        b, g, r = self.colour_bgr
        return (r, g, b)

    @property
    def colour_hex(self) -> str:
        r, g, b = self.colour_rgb
        return f"#{r:02x}{g:02x}{b:02x}"


def detect_contours(
    binary: np.ndarray,
    min_area: float = 100.0,
    retrieval_mode: int = cv2.RETR_EXTERNAL,
) -> list[np.ndarray]:
    if binary is None or binary.ndim != 2:
        logger.error("detect_contours: expected 2D binary array.")
        return []

    try:
        contours, _ = cv2.findContours(
            binary, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE
        )
        filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
        filtered.sort(key=cv2.contourArea, reverse=True)
        logger.info(
            "detect_contours: found %d contours (≥%.0f px²) from %d total",
            len(filtered), min_area, len(contours),
        )
        return filtered
    except Exception as exc:
        logger.exception("detect_contours failed: %s", exc)
        return []


def measure_contour(contour: np.ndarray, idx: int) -> ContourInfo:
    info = ContourInfo(
        index=idx,
        colour_bgr=_get_colour(idx),
        points=contour,
    )

    info.area = cv2.contourArea(contour)
    info.perimeter = cv2.arcLength(contour, closed=True)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        info.cx = int(M["m10"] / M["m00"])
        info.cy = int(M["m01"] / M["m00"])

    x, y, w, h = cv2.boundingRect(contour)
    info.bbox_x, info.bbox_y, info.bbox_w, info.bbox_h = x, y, w, h

    info.aspect_ratio = round(w / h, 3) if h > 0 else 0.0

    bbox_area = w * h
    info.extent = round(info.area / bbox_area, 3) if bbox_area > 0 else 0.0

    if info.perimeter > 0:
        info.circularity = round(
            4 * np.pi * info.area / (info.perimeter ** 2), 3
        )

    return info


def draw_contours_overlay(
    rgb_image: np.ndarray,
    contour_infos: list[ContourInfo],
    show_contours:   bool = True,
    show_bbox:       bool = True,
    show_centroid:   bool = True,
    show_labels:     bool = True,
    contour_thickness: int = 2,
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR).copy()

    for info in contour_infos:
        colour = info.colour_bgr

        if show_contours:
            cv2.drawContours(
                canvas, [info.points], -1, colour, contour_thickness
            )

        if show_bbox:
            cv2.rectangle(
                canvas,
                (info.bbox_x, info.bbox_y),
                (info.bbox_x + info.bbox_w, info.bbox_y + info.bbox_h),
                colour, 1,
            )

        if show_centroid:
            cv2.circle(canvas, (info.cx, info.cy), 6, (255, 255, 255), -1)
            cv2.circle(canvas, (info.cx, info.cy), 4, colour, -1)
            arm = 12
            cv2.line(canvas, (info.cx - arm, info.cy), (info.cx + arm, info.cy), colour, 1)
            cv2.line(canvas, (info.cx, info.cy - arm), (info.cx, info.cy + arm), colour, 1)

        if show_labels:
            label = f"#{info.index + 1}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx, ty = info.bbox_x + 3, info.bbox_y + th + 4
            cv2.rectangle(canvas, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2),
                          (0, 0, 0), -1)
            cv2.putText(canvas, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def run_contour_analysis(
    rgb_image: np.ndarray,
    binary: np.ndarray,
    min_area: float = 100.0,
    show_contours: bool = True,
    show_bbox: bool = True,
    show_centroid: bool = True,
    show_labels: bool = True,
    contour_thickness: int = 2,
    max_contours: int = 20,
) -> dict:
    raw = detect_contours(binary, min_area=min_area)
    raw = raw[:max_contours]

    infos: list[ContourInfo] = [measure_contour(c, i) for i, c in enumerate(raw)]

    overlay: Optional[np.ndarray] = None
    if rgb_image is not None and infos:
        overlay = draw_contours_overlay(
            rgb_image, infos,
            show_contours=show_contours,
            show_bbox=show_bbox,
            show_centroid=show_centroid,
            show_labels=show_labels,
            contour_thickness=contour_thickness,
        )
    elif rgb_image is not None:
        overlay = rgb_image.copy()

    total_area = sum(c.area for c in infos)

    logger.info(
        "run_contour_analysis: %d contours, total_area=%.0f px²",
        len(infos), total_area,
    )

    return {
        "contours":   infos,
        "overlay":    overlay,
        "count":      len(infos),
        "total_area": total_area,
    }