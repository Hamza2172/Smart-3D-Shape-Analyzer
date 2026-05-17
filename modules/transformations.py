import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _homogenise_2d(pts: np.ndarray) -> np.ndarray:
    return np.hstack([pts, np.ones((len(pts), 1))])


def apply_transformation_matrix(
    points: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    h = _homogenise_2d(points.astype(np.float64))
    transformed = (matrix @ h.T).T
    return transformed[:, :2]


def translate_points(
    points: np.ndarray,
    tx: float = 0.0,
    ty: float = 0.0,
) -> np.ndarray:
    M = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1 ],
    ], dtype=np.float64)
    return apply_transformation_matrix(points, M)


def scale_points(
    points: np.ndarray,
    sx: float = 1.0,
    sy: Optional[float] = None,
    origin: Optional[tuple] = None,
) -> np.ndarray:
    sy = sy if sy is not None else sx
    if origin is None:
        ox, oy = points[:, 0].mean(), points[:, 1].mean()
    else:
        ox, oy = origin

    T_to   = np.array([[1,0,-ox],[0,1,-oy],[0,0,1]], dtype=np.float64)
    S      = np.array([[sx,0,0],[0,sy,0],[0,0,1]],   dtype=np.float64)
    T_back = np.array([[1,0, ox],[0,1, oy],[0,0,1]], dtype=np.float64)
    M = T_back @ S @ T_to
    return apply_transformation_matrix(points, M)


def rotate_points(
    points: np.ndarray,
    angle_deg: float = 0.0,
    origin: Optional[tuple] = None,
) -> np.ndarray:
    if origin is None:
        ox, oy = points[:, 0].mean(), points[:, 1].mean()
    else:
        ox, oy = origin

    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    T_to   = np.array([[1,0,-ox],[0,1,-oy],[0,0,1]], dtype=np.float64)
    R      = np.array([[cos_a,-sin_a,0],[sin_a,cos_a,0],[0,0,1]], dtype=np.float64)
    T_back = np.array([[1,0, ox],[0,1, oy],[0,0,1]], dtype=np.float64)
    M = T_back @ R @ T_to
    return apply_transformation_matrix(points, M)


def reflect_points(
    points: np.ndarray,
    axis: str = "horizontal",
    origin: Optional[tuple] = None,
) -> np.ndarray:
    if origin is None:
        ox, oy = points[:, 0].mean(), points[:, 1].mean()
    else:
        ox, oy = origin

    sx = -1.0 if axis in ("vertical", "both") else 1.0
    sy = -1.0 if axis in ("horizontal", "both") else 1.0

    T_to   = np.array([[1,0,-ox],[0,1,-oy],[0,0,1]], dtype=np.float64)
    Rf     = np.array([[sx,0,0],[0,sy,0],[0,0,1]],   dtype=np.float64)
    T_back = np.array([[1,0, ox],[0,1, oy],[0,0,1]], dtype=np.float64)
    M = T_back @ Rf @ T_to
    return apply_transformation_matrix(points, M)


def shear_points(
    points: np.ndarray,
    shx: float = 0.0,
    shy: float = 0.0,
) -> np.ndarray:
    M = np.array([
        [1,  shx, 0],
        [shy, 1,  0],
        [0,   0,  1],
    ], dtype=np.float64)
    return apply_transformation_matrix(points, M)


def rotate_points_3d(
    points: np.ndarray,
    rx: float = 0.0,
    ry: float = 0.0,
    rz: float = 0.0,
) -> np.ndarray:
    if points is None or len(points) == 0:
        return points

    def _Rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])

    def _Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])

    def _Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    R = _Rz(np.deg2rad(rz)) @ _Ry(np.deg2rad(ry)) @ _Rx(np.deg2rad(rx))
    centre = points.mean(axis=0)
    return ((points - centre) @ R.T) + centre


def scale_points_3d(
    points: np.ndarray,
    sx: float = 1.0,
    sy: float = 1.0,
    sz: float = 1.0,
) -> np.ndarray:
    if points is None or len(points) == 0:
        return points
    centre = points.mean(axis=0)
    S = np.diag([sx, sy, sz])
    return ((points - centre) @ S) + centre


def translate_points_3d(
    points: np.ndarray,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
) -> np.ndarray:
    if points is None or len(points) == 0:
        return points
    return points + np.array([tx, ty, tz])


def apply_2d_transforms(
    points: np.ndarray,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
    reflect_axis: str = "none",
    shx: float = 0.0,
    shy: float = 0.0,
) -> np.ndarray:
    pts = points.astype(np.float64)

    if tx != 0.0 or ty != 0.0:
        pts = translate_points(pts, tx, ty)
    if scale != 1.0:
        pts = scale_points(pts, scale)
    if angle_deg != 0.0:
        pts = rotate_points(pts, angle_deg)
    if reflect_axis != "none":
        pts = reflect_points(pts, axis=reflect_axis)
    if shx != 0.0 or shy != 0.0:
        pts = shear_points(pts, shx, shy)

    return pts