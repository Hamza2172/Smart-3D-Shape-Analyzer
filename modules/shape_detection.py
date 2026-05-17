import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

from modules.contour_analysis import ContourInfo

logger = logging.getLogger(__name__)


@dataclass
class ShapeInfo:
    contour_index:  int
    shape_name:     str
    vertices_count: int
    confidence:     float
    contour_area:   float


_APPROX_EPSILON_FACTOR: float = 0.03
_CIRCLE_CIRCULARITY_MIN: float = 0.80
_SQUARE_ASPECT_MIN: float = 0.90
_SQUARE_ASPECT_MAX: float = 1.10


def detect_shape(info: ContourInfo) -> ShapeInfo:
    contour   = info.points
    perimeter = info.perimeter

    epsilon    = _APPROX_EPSILON_FACTOR * perimeter
    approx     = cv2.approxPolyDP(contour, epsilon, closed=True)
    n_vertices = len(approx)

    shape_name: str   = "Unknown"
    confidence: float = 0.0

    if info.circularity >= _CIRCLE_CIRCULARITY_MIN:
        shape_name = "Circle"
        confidence = round(
            (info.circularity - _CIRCLE_CIRCULARITY_MIN) / (1.0 - _CIRCLE_CIRCULARITY_MIN),
            3,
        )
        confidence = min(confidence, 1.0)

    elif n_vertices == 3:
        shape_name = "Triangle"
        confidence = _poly_confidence(info.circularity, target=0.60)

    elif n_vertices == 4:
        ar = info.aspect_ratio
        if _SQUARE_ASPECT_MIN <= ar <= _SQUARE_ASPECT_MAX:
            shape_name = "Square"
            deviation  = abs(ar - 1.0)
            confidence = round(max(0.0, 1.0 - deviation / 0.10), 3)
        else:
            shape_name = "Rectangle"
            confidence = _poly_confidence(info.extent, target=0.80)

    elif n_vertices == 5:
        shape_name = "Pentagon"
        confidence = _poly_confidence(info.circularity, target=0.72)

    elif n_vertices == 6:
        shape_name = "Hexagon"
        confidence = _poly_confidence(info.circularity, target=0.75)

    elif n_vertices == 7:
        shape_name = "Heptagon"
        confidence = _poly_confidence(info.circularity, target=0.78)

    elif n_vertices == 8:
        shape_name = "Octagon"
        confidence = _poly_confidence(info.circularity, target=0.82)

    elif n_vertices > 8:
        if info.circularity >= 0.70:
            shape_name = "Circle"
            confidence = round(min(info.circularity, 1.0), 3)
        else:
            shape_name = f"Polygon ({n_vertices}v)"
            confidence = 0.50

    return ShapeInfo(
        contour_index=info.index,
        shape_name=shape_name,
        vertices_count=n_vertices,
        confidence=confidence,
        contour_area=info.area,
    )


def _poly_confidence(metric: float, target: float) -> float:
    if target <= 0:
        return 0.0
    deviation = abs(metric - target)
    raw = max(0.0, 1.0 - deviation / target)
    return round(raw, 3)


def analyze_shapes(contour_infos: list[ContourInfo]) -> list[ShapeInfo]:
    results = []
    for info in contour_infos:
        try:
            shape = detect_shape(info)
            results.append(shape)
            logger.info(
                "Shape #%d → %s  (vertices=%d  confidence=%.2f  area=%.0f)",
                info.index + 1, shape.shape_name,
                shape.vertices_count, shape.confidence, shape.contour_area,
            )
        except Exception as exc:
            logger.exception("detect_shape failed for contour %d: %s", info.index, exc)
            results.append(ShapeInfo(
                contour_index=info.index,
                shape_name="Error",
                vertices_count=0,
                confidence=0.0,
                contour_area=info.area,
            ))
    return results


def draw_shape_labels(
    rgb_image: np.ndarray,
    contour_infos: list[ContourInfo],
    shape_infos: list[ShapeInfo],
    font_scale: float = 0.5,
) -> np.ndarray:
    if rgb_image is None:
        return rgb_image

    c_map: dict[int, ContourInfo] = {c.index: c for c in contour_infos}

    canvas = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR).copy()
    font   = cv2.FONT_HERSHEY_SIMPLEX
    lh     = int(20 * font_scale)

    for s in shape_infos:
        ci = c_map.get(s.contour_index)
        if ci is None:
            continue

        colour = ci.colour_bgr
        tx = ci.bbox_x
        ty = max(ci.bbox_y - lh - 4, lh + 4)

        name_label   = s.shape_name
        detail_label = f"{s.vertices_count}v  {s.confidence:.0%}"

        for i, label in enumerate([name_label, detail_label]):
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            ly = ty + i * (th + 4)
            cv2.rectangle(canvas,
                          (tx - 2, ly - th - 2),
                          (tx + tw + 2, ly + 2),
                          (0, 0, 0), -1)
            cv2.putText(canvas, label, (tx, ly),
                        font, font_scale, colour, 1, cv2.LINE_AA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)