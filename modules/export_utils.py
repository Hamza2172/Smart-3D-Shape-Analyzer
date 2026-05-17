import os
import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_export_filename(label: str, extension: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{ts}.{extension}"
    return os.path.join(OUTPUT_DIR, filename)


def save_image(
    image: np.ndarray,
    label: str = "image",
    filepath: Optional[str] = None,
) -> tuple[bool, str]:
    if image is None:
        logger.error("save_image: received None image.")
        return False, ""

    path = filepath or generate_export_filename(label, "png")
    try:
        if image.ndim == 3 and image.shape[2] == 3:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr = image
        cv2.imwrite(path, bgr)
        logger.info("Saved image → %s", path)
        return True, path
    except Exception as exc:
        logger.exception("save_image failed: %s", exc)
        return False, ""


def save_pointcloud_ply(
    points: np.ndarray,
    label: str = "pointcloud",
    filepath: Optional[str] = None,
) -> tuple[bool, str]:
    if points is None or len(points) == 0:
        logger.error("save_pointcloud_ply: empty point array.")
        return False, ""

    path = filepath or generate_export_filename(label, "ply")

    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        o3d.io.write_point_cloud(path, pcd)
        logger.info("Saved PLY (Open3D) → %s", path)
        return True, path
    except ImportError:
        pass
    except Exception as exc:
        logger.exception("Open3D PLY write failed: %s", exc)

    try:
        n = len(points)
        header = (
            "ply\nformat ascii 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        )
        with open(path, "w") as f:
            f.write(header)
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        logger.info("Saved PLY (ASCII fallback) → %s", path)
        return True, path
    except Exception as exc:
        logger.exception("ASCII PLY write failed: %s", exc)
        return False, ""


def save_pointcloud_obj(
    points: np.ndarray,
    label: str = "pointcloud",
    filepath: Optional[str] = None,
) -> tuple[bool, str]:
    if points is None or len(points) == 0:
        return False, ""

    path = filepath or generate_export_filename(label, "obj")
    try:
        with open(path, "w") as f:
            for p in points:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        logger.info("Saved OBJ → %s", path)
        return True, path
    except Exception as exc:
        logger.exception("save_pointcloud_obj failed: %s", exc)
        return False, ""


def image_to_png_bytes(image: np.ndarray) -> Optional[bytes]:
    if image is None:
        return None
    try:
        if image.ndim == 3 and image.shape[2] == 3:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr = image
        ok, buf = cv2.imencode(".png", bgr)
        return bytes(buf) if ok else None
    except Exception as exc:
        logger.exception("image_to_png_bytes failed: %s", exc)
        return None


def pointcloud_to_ply_bytes(points: np.ndarray) -> Optional[bytes]:
    if points is None or len(points) == 0:
        return None
    try:
        n = len(points)
        lines = [
            "ply", "format ascii 1.0",
            f"element vertex {n}",
            "property float x", "property float y", "property float z",
            "end_header",
        ]
        lines += [f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in points]
        return "\n".join(lines).encode("utf-8")
    except Exception as exc:
        logger.exception("pointcloud_to_ply_bytes failed: %s", exc)
        return None