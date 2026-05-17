import numpy as np
from typing import Optional
import logging

from modules.contour_analysis import ContourInfo

logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False
    logger.warning("open3d not installed — Stage 7 mesh features disabled.")

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    logger.warning("plotly not installed — Stage 7 visualisation disabled.")


def contour_to_pointcloud(
    contour_info: ContourInfo,
    z_value: float = 0.0,
    image_height: int = 1,
) -> np.ndarray:
    pts = contour_info.points.reshape(-1, 2).astype(np.float64)
    scale = float(image_height) if image_height > 0 else 1.0
    x = pts[:, 0] / scale
    y = 1.0 - pts[:, 1] / scale
    z = np.full(len(pts), z_value)
    return np.column_stack([x, y, z])


def build_pointcloud_from_contours(
    contour_infos: list[ContourInfo],
    image_height: int = 512,
) -> np.ndarray:
    if not contour_infos:
        return np.zeros((0, 3))
    clouds = [contour_to_pointcloud(c, z_value=0.0, image_height=image_height)
              for c in contour_infos]
    return np.vstack(clouds)


def extrude_contour(
    contour_info: ContourInfo,
    depth: float = 0.2,
    image_height: int = 512,
) -> np.ndarray:
    base = contour_to_pointcloud(contour_info, z_value=0.0, image_height=image_height)
    top  = base.copy()
    top[:, 2] = depth
    return np.vstack([base, top])


def build_extruded_pointcloud(
    contour_infos: list[ContourInfo],
    depth: float = 0.2,
    image_height: int = 512,
) -> np.ndarray:
    if not contour_infos:
        return np.zeros((0, 3))
    clouds = [extrude_contour(c, depth=depth, image_height=image_height)
              for c in contour_infos]
    return np.vstack(clouds)


def create_o3d_pointcloud(points: np.ndarray, colour: tuple = (0.2, 0.7, 1.0)):
    if not _O3D_AVAILABLE or points is None or len(points) == 0:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(list(colour))
    return pcd


def save_o3d_pointcloud(pcd, filepath: str) -> bool:
    if not _O3D_AVAILABLE or pcd is None:
        return False
    try:
        o3d.io.write_point_cloud(filepath, pcd)
        logger.info("Saved point cloud → %s", filepath)
        return True
    except Exception as exc:
        logger.exception("save_o3d_pointcloud failed: %s", exc)
        return False


def render_3d_figure(
    contour_infos: list[ContourInfo],
    image_height: int = 512,
    extrude: bool = False,
    depth: float = 0.2,
    point_size: int = 3,
    render_mode: str = "points",
):
    if not _PLOTLY_AVAILABLE:
        logger.error("render_3d_figure: plotly not available.")
        return None

    fig = go.Figure()

    for ci in contour_infos:
        r, g, b = ci.colour_rgb
        colour_str = f"rgb({r},{g},{b})"

        base = contour_to_pointcloud(ci, z_value=0.0, image_height=image_height)
        xs, ys, zs = base[:, 0], base[:, 1], base[:, 2]

        if extrude:
            top = base.copy(); top[:, 2] = depth
            for i in range(len(base)):
                fig.add_trace(go.Scatter3d(
                    x=[base[i, 0], top[i, 0]],
                    y=[base[i, 1], top[i, 1]],
                    z=[base[i, 2], top[i, 2]],
                    mode="lines",
                    line=dict(color=colour_str, width=1),
                    showlegend=False,
                    hoverinfo="skip",
                ))
            fig.add_trace(go.Scatter3d(
                x=list(top[:, 0]) + [top[0, 0]],
                y=list(top[:, 1]) + [top[0, 1]],
                z=list(top[:, 2]) + [top[0, 2]],
                mode="lines",
                line=dict(color=colour_str, width=2),
                name=f"Top #{ci.index + 1}",
            ))

        if render_mode == "points":
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                marker=dict(size=point_size, color=colour_str, opacity=0.85),
                name=f"Contour #{ci.index + 1}",
            ))
        elif render_mode == "lines":
            xs_c = np.append(xs, xs[0]); ys_c = np.append(ys, ys[0]); zs_c = np.append(zs, zs[0])
            fig.add_trace(go.Scatter3d(
                x=xs_c, y=ys_c, z=zs_c,
                mode="lines",
                line=dict(color=colour_str, width=3),
                name=f"Contour #{ci.index + 1}",
            ))
        elif render_mode == "surface":
            cx_n = ci.cx / float(image_height)
            cy_n = 1.0 - ci.cy / float(image_height)
            vx = np.append(xs, cx_n)
            vy = np.append(ys, cy_n)
            vz = np.append(zs, 0.0)
            n  = len(xs)
            i_idx = list(range(n))
            j_idx = list(range(1, n)) + [0]
            k_idx = [n] * n
            fig.add_trace(go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=i_idx, j=j_idx, k=k_idx,
                color=colour_str, opacity=0.5,
                name=f"Contour #{ci.index + 1}",
            ))

    fig.update_layout(
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#0d0f14",
        scene=dict(
            bgcolor="#0d0f14",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=True,  title="Z"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(font=dict(color="#c8ccd8", size=11), bgcolor="#11141c"),
        height=480,
    )
    return fig