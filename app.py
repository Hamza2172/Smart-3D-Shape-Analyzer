import logging
from typing import Optional
import numpy as np
import streamlit as st
from modules.image_loader import load_image_from_bytes, get_image_metadata
from modules.preprocessing import run_preprocessing_pipeline
from modules.morphology import run_morphology, OPERATIONS, OPERATION_DESCRIPTIONS
from modules.contour_analysis import run_contour_analysis, ContourInfo
from modules.shape_detection import analyze_shapes, draw_shape_labels, ShapeInfo
from modules.skeleton import run_skeleton_pipeline, _SKIMAGE_AVAILABLE
from modules.mesh_builder import render_3d_figure, build_pointcloud_from_contours, build_extruded_pointcloud
from modules.transformations import apply_2d_transforms, rotate_points_3d, scale_points_3d
from modules.export_utils import image_to_png_bytes, pointcloud_to_ply_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Smart 3D Shape Analyzer",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace;
        background-color: #0d0f14;
        color: #c8ccd8;
    }

    .block-container {
        padding: 2rem 2.5rem;
        max-width: 1280px;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.6rem;
        letter-spacing: -0.03em;
        background: linear-gradient(120deg, #e0e8ff 0%, #7eb8f7 50%, #4f8ef7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
        margin-bottom: 0.2rem;
    }

    .hero-stage {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #4f8ef7;
        margin-bottom: 1.2rem;
    }

    .hero-desc {
        font-size: 0.9rem;
        color: #7a8097;
        max-width: 560px;
        line-height: 1.7;
        margin-bottom: 2rem;
    }

    hr {
        border: none;
        border-top: 1px solid #1e2230;
        margin: 1.5rem 0;
    }

    .image-panel {
        background: #11141c;
        border: 1px solid #1e2230;
        border-radius: 12px;
        padding: 1.5rem;
    }

    .panel-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #4f8ef7;
        margin-bottom: 0.8rem;
    }

    .meta-card {
        background: #0d0f14;
        border: 1px solid #1e2230;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }

    .meta-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid #1a1d27;
        font-size: 0.82rem;
    }

    .meta-row:last-child { border-bottom: none; }

    .meta-key { color: #555c78; }
    .meta-val { color: #c8ccd8; font-weight: 500; }

    [data-testid="stFileUploader"] {
        background: #11141c;
        border: 1px dashed #2a3050;
        border-radius: 10px;
        padding: 0.5rem;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #4f8ef7;
    }

    [data-testid="stSidebar"] {
        background: #0b0d12;
        border-right: 1px solid #1e2230;
    }

    .sidebar-section {
        margin-bottom: 1.6rem;
    }

    .sidebar-heading {
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #4f8ef7;
        margin-bottom: 0.6rem;
    }

    .stage-pill {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.73rem;
        margin: 0.2rem 0;
        border: 1px solid;
    }

    .stage-active {
        background: #0e1e3d;
        border-color: #4f8ef7;
        color: #7eb8f7;
    }

    .stage-upcoming {
        background: #111318;
        border-color: #2a2f3d;
        color: #3a3f52;
    }

    .placeholder-box {
        background: #11141c;
        border: 1px dashed #252a38;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        color: #3a4060;
        font-size: 0.85rem;
    }

    .info-box {
        background: #0b1626;
        border-left: 3px solid #4f8ef7;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #7eb8f7;
        margin-bottom: 1rem;
    }

    .err-box {
        background: #1a0d0d;
        border-left: 3px solid #f74f4f;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #f78888;
        margin-bottom: 1rem;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-section">
                <div class="sidebar-heading">Project</div>
                <span style="font-family:'Syne',sans-serif;font-size:1rem;
                             font-weight:700;color:#c8ccd8;">
                    Smart 3D Shape Analyzer
                </span><br>
                <span style="font-size:0.75rem;color:#555c78;">v1.0.0 — Full Pipeline</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-heading">Roadmap</div>', unsafe_allow_html=True)

        stages = [
            ("01", "Image Upload & Display",       True),
            ("02", "Preprocessing & Enhancement",  True),
            ("03", "Morphological Operations",     True),
            ("04", "Contour Detection & Metrics",  True),
            ("05", "Shape Detection",              True),
            ("06", "Skeletonization & Thinning",   True),
            ("07", "3D Mesh & Point Cloud",        True),
            ("08", "Geometric Transformations",    True),
            ("09", "Export Results",               True),
        ]

        for num, label, active in stages:
            pill_cls = "stage-active" if active else "stage-upcoming"
            st.markdown(
                f"""
                <div class="stage-pill {pill_cls}">
                    <b>Stage {num}</b> — {label}
                </div><br>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-heading">Upload Settings</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div style="font-size:0.78rem;color:#555c78;line-height:1.8;">
                Accepted formats<br>
                <span style="color:#7eb8f7;">JPG &nbsp;·&nbsp; PNG &nbsp;·&nbsp; BMP &nbsp;·&nbsp; TIFF &nbsp;·&nbsp; WEBP</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-heading">⚙ Preprocessing Controls</div>', unsafe_allow_html=True)

        blur_kernel = st.slider(
            "Gaussian Blur — kernel size",
            min_value=1, max_value=21, value=5, step=2,
            help="Controls blur strength. Must be odd; even values are auto-corrected.",
        )

        st.markdown('<div class="sidebar-heading" style="margin-top:0.8rem;">Thresholding</div>', unsafe_allow_html=True)

        use_adaptive = st.toggle(
            "Use Adaptive Threshold",
            value=False,
            help="Adaptive computes a local threshold per region — better for uneven lighting.",
        )

        if use_adaptive:
            adaptive_block = st.slider(
                "Block size (neighbourhood)",
                min_value=3, max_value=51, value=11, step=2,
                help="Pixel neighbourhood for local mean. Larger = smoother regions.",
            )
            adaptive_c = st.slider(
                "Constant C",
                min_value=0, max_value=20, value=2,
                help="Subtracted from local mean. Higher = less foreground.",
            )
            threshold_value = 127
        else:
            threshold_value = st.slider(
                "Global threshold value",
                min_value=0, max_value=255, value=127,
                help="Pixels above this value become white (foreground).",
            )
            adaptive_block = 11
            adaptive_c = 2

        invert = st.toggle(
            "Invert binary output",
            value=False,
            help="Swap foreground/background — useful for dark objects on light background.",
        )

        denoise_kernel = st.slider(
            "Noise reduction — kernel size",
            min_value=1, max_value=9, value=3, step=2,
            help="Median filter kernel. Larger = stronger noise suppression.",
        )

        st.session_state["preproc_params"] = {
            "blur_kernel": blur_kernel,
            "threshold_value": threshold_value,
            "use_adaptive": use_adaptive,
            "adaptive_block_size": adaptive_block,
            "adaptive_c": adaptive_c,
            "invert_threshold": invert,
            "denoise_kernel": denoise_kernel,
        }

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-heading">🔷 Morphology Controls</div>',
            unsafe_allow_html=True,
        )

        morph_operation = st.selectbox(
            "Operation",
            options=list(OPERATIONS.keys()),
            index=0,
            help="Select the morphological operation to apply to the binary image.",
        )

        morph_kernel = st.slider(
            "Kernel size",
            min_value=1, max_value=21, value=3, step=2,
            help="Size of the structuring element. Larger = stronger effect.",
        )

        morph_iterations = st.slider(
            "Iterations",
            min_value=1, max_value=10, value=1,
            help="Number of times the operation is applied in sequence.",
        )

        st.session_state["morph_params"] = {
            "operation":   morph_operation,
            "kernel_size": morph_kernel,
            "iterations":  morph_iterations,
        }

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-heading">🔶 Contour Controls</div>',
            unsafe_allow_html=True,
        )

        min_area = st.slider(
            "Min contour area (px²)",
            min_value=10, max_value=5000, value=100, step=10,
            help="Contours smaller than this area are discarded as noise.",
        )

        max_contours = st.slider(
            "Max contours to analyse",
            min_value=1, max_value=50, value=20,
            help="Cap on number of contours processed (largest first).",
        )

        contour_thickness = st.slider(
            "Outline thickness (px)",
            min_value=1, max_value=8, value=2,
            help="Pixel width of the drawn contour lines.",
        )

        st.markdown(
            '<div class="sidebar-heading" style="margin-top:0.8rem;">Overlay Toggles</div>',
            unsafe_allow_html=True,
        )
        show_contours = st.toggle("Show contour outlines", value=True)
        show_bbox     = st.toggle("Show bounding boxes",   value=True)
        show_centroid = st.toggle("Show centroid markers", value=True)
        show_labels   = st.toggle("Show index labels",     value=True)

        st.session_state["contour_params"] = {
            "min_area":          min_area,
            "max_contours":      max_contours,
            "contour_thickness": contour_thickness,
            "show_contours":     show_contours,
            "show_bbox":         show_bbox,
            "show_centroid":     show_centroid,
            "show_labels":       show_labels,
        }

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-heading">🔺 Shape Detection</div>',
            unsafe_allow_html=True,
        )
        enable_shape_detection = st.toggle(
            "Enable shape detection", value=True,
            help="Classify each contour into Triangle, Circle, Rectangle, etc.",
        )
        show_shape_labels = st.toggle(
            "Show labels on overlay", value=True,
            help="Draw shape name and vertex count on the contour overlay.",
        )
        st.session_state["shape_params"] = {
            "enabled":    enable_shape_detection,
            "show_labels": show_shape_labels,
        }

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-heading">🦴 Skeletonization</div>',
            unsafe_allow_html=True,
        )
        if not _SKIMAGE_AVAILABLE:
            st.markdown(
                '<div style="font-size:0.75rem;color:#f78888;">'
                "scikit-image not installed.<br>"
                "<code>pip install scikit-image</code></div>",
                unsafe_allow_html=True,
            )
        enable_skeleton = st.toggle(
            "Enable skeletonization", value=_SKIMAGE_AVAILABLE,
            disabled=not _SKIMAGE_AVAILABLE,
            help="Reduce binary shapes to 1-pixel-wide skeletons.",
        )
        skel_mode = st.selectbox(
            "Mode",
            options=["skeleton", "thinning"],
            format_func=lambda x: "Skeleton (medial axis)" if x == "skeleton" else "Thinning (Guo-Hall)",
            help="Skeleton: medial axis. Thinning: more conservative, preserves connectivity.",
        )
        skel_thickness = st.slider(
            "Overlay line thickness",
            min_value=1, max_value=4, value=1,
            help="Visual dilation applied to skeleton pixels in the overlay.",
        )
        st.session_state["skeleton_params"] = {
            "enabled":           enable_skeleton,
            "mode":              skel_mode,
            "overlay_thickness": skel_thickness,
        }

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-heading">🧊 3D Visualisation</div>', unsafe_allow_html=True)
        enable_3d = st.toggle("Enable 3D view", value=True)
        render_mode_3d = st.selectbox(
            "Render mode",
            options=["points", "lines", "surface"],
            format_func=lambda x: {"points": "Point Cloud", "lines": "Wire Frame", "surface": "Filled Surface"}[x],
        )
        enable_extrude = st.toggle("Extrude contours", value=False,
                                   help="Add Z-depth to create 3D shells.")
        extrude_depth = st.slider("Extrusion depth", 0.05, 1.0, 0.2, 0.05,
                                  disabled=not enable_extrude)
        point_size_3d = st.slider("Point size", 1, 10, 3)
        st.session_state["mesh_params"] = {
            "enabled":      enable_3d,
            "render_mode":  render_mode_3d,
            "extrude":      enable_extrude,
            "depth":        extrude_depth,
            "point_size":   point_size_3d,
        }

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-heading">🔄 Transformations (2D)</div>', unsafe_allow_html=True)
        enable_transform = st.toggle("Enable transformations", value=False)
        t_angle = st.slider("Rotation (°)",    -180, 180, 0,   disabled=not enable_transform)
        t_scale = st.slider("Scale",           0.1,  3.0, 1.0, step=0.05, disabled=not enable_transform)
        t_tx    = st.slider("Translate X",    -200,  200, 0,   disabled=not enable_transform)
        t_ty    = st.slider("Translate Y",    -200,  200, 0,   disabled=not enable_transform)
        t_ref   = st.selectbox("Reflection",
                               ["none", "horizontal", "vertical", "both"],
                               disabled=not enable_transform)
        t_shx   = st.slider("Shear X", -1.0, 1.0, 0.0, step=0.05, disabled=not enable_transform)
        st.session_state["transform_params"] = {
            "enabled":      enable_transform,
            "angle_deg":    float(t_angle),
            "scale":        t_scale,
            "tx":           float(t_tx),
            "ty":           float(t_ty),
            "reflect_axis": t_ref,
            "shx":          t_shx,
            "shy":          0.0,
        }

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.7rem;color:#2a3050;text-align:center;">'
            "Built with OpenCV · scikit-image · Plotly · Streamlit<br>© 2025 Smart 3D Shape Analyzer"
            "</div>",
            unsafe_allow_html=True,
        )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-stage">🔷 &nbsp; Complete Computer Vision Pipeline — v1.0</div>
        <div class="hero-title">Smart 3D Shape Analyzer</div>
        <div class="hero-desc">
            Full end-to-end pipeline: upload → preprocess → detect → classify →
            skeletonize → visualize in 3D → transform → export. All stages are
            live and configurable from the sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)


def render_image_panel(image, metadata: dict) -> None:
    col_img, col_meta = st.columns([3, 1], gap="large")

    with col_img:
        st.markdown('<div class="panel-label">📷 &nbsp; Uploaded Image</div>', unsafe_allow_html=True)
        st.image(image, width="stretch")

    with col_meta:
        st.markdown('<div class="panel-label">📊 &nbsp; Image Metadata</div>', unsafe_allow_html=True)

        meta_rows = [
            ("Width", f"{metadata['width']} px"),
            ("Height", f"{metadata['height']} px"),
            ("Channels", str(metadata["channels"])),
            ("Dtype", metadata["dtype"]),
            ("Aspect Ratio", str(metadata["aspect_ratio"])),
        ]

        rows_html = "".join(
            f"""
            <div class="meta-row">
                <span class="meta-key">{key}</span>
                <span class="meta-val">{val}</span>
            </div>
            """
            for key, val in meta_rows
        )
        st.markdown(
            f'<div class="meta-card">{rows_html}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
                ✅ &nbsp; Stage 1 complete.<br>
                Image ready for preprocessing.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_preprocessing_panel(rgb_image: np.ndarray, params: dict) -> None:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-label">🔬 &nbsp; Stage 2 — Preprocessing Pipeline</div>',
        unsafe_allow_html=True,
    )

    results = run_preprocessing_pipeline(rgb_image, **params)

    method_label = "Adaptive Threshold" if params["use_adaptive"] else "Global Threshold"

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown('<div class="panel-label">① Original (RGB)</div>', unsafe_allow_html=True)
        st.image(rgb_image, width="stretch")

    with col2:
        st.markdown('<div class="panel-label">② Grayscale</div>', unsafe_allow_html=True)
        if results["grayscale"] is not None:
            st.image(results["grayscale"], width="stretch", clamp=True)
        else:
            st.markdown('<div class="err-box">Grayscale conversion failed.</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="panel-label">③ Noise Reduction (Median)</div>', unsafe_allow_html=True)
        if results["denoised"] is not None:
            st.image(results["denoised"], width="stretch", clamp=True)
        else:
            st.markdown('<div class="err-box">Denoising failed.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3, gap="medium")

    with col4:
        st.markdown('<div class="panel-label">④ Gaussian Blur</div>', unsafe_allow_html=True)
        if results["blurred"] is not None:
            st.image(results["blurred"], width="stretch", clamp=True)
        else:
            st.markdown('<div class="err-box">Blur step failed.</div>', unsafe_allow_html=True)

    with col5:
        st.markdown(
            f'<div class="panel-label">⑤ Binary — {method_label}</div>',
            unsafe_allow_html=True,
        )
        if results["binary"] is not None:
            st.image(results["binary"], width="stretch", clamp=True)
        else:
            st.markdown('<div class="err-box">Thresholding failed.</div>', unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="panel-label">📈 &nbsp; Pipeline Stats</div>', unsafe_allow_html=True)

        binary = results["binary"]
        if binary is not None:
            total_px = binary.size
            white_px = int(np.sum(binary == 255))
            black_px = total_px - white_px
            fg_pct   = round(white_px / total_px * 100, 1)
            bg_pct   = round(black_px / total_px * 100, 1)

            stat_rows = [
                ("Method",      method_label),
                ("Blur kernel", f"{params['blur_kernel']}×{params['blur_kernel']}"),
                ("Invert",      "Yes" if params["invert_threshold"] else "No"),
                ("Foreground",  f"{fg_pct}%"),
                ("Background",  f"{bg_pct}%"),
                ("Total px",    f"{total_px:,}"),
            ]
        else:
            stat_rows = [("Status", "Pipeline incomplete")]

        rows_html = "".join(
            f"""
            <div class="meta-row">
                <span class="meta-key">{k}</span>
                <span class="meta-val">{v}</span>
            </div>
            """
            for k, v in stat_rows
        )
        st.markdown(f'<div class="meta-card">{rows_html}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
                ✅ &nbsp; Stage 2 complete.<br>
                Binary image passed to Stage 3.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_morphology_panel(binary: np.ndarray, params: dict) -> None:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-label">🔷 &nbsp; Stage 3 — Morphological Operations</div>',
        unsafe_allow_html=True,
    )

    operation   = params["operation"]
    kernel_size = params["kernel_size"]
    iterations  = params["iterations"]

    desc = OPERATION_DESCRIPTIONS.get(operation, "")
    st.markdown(
        f"""
        <div style="background:#0d1520;border-left:3px solid #4f8ef7;
                    border-radius:6px;padding:0.65rem 1rem;margin-bottom:1rem;
                    font-size:0.82rem;color:#7eb8f7;line-height:1.6;">
            <b>{operation}</b> &nbsp;—&nbsp; {desc}
        </div>
        """,
        unsafe_allow_html=True,
    )

    out = run_morphology(binary, operation, kernel_size=kernel_size, iterations=iterations)

    result         = out["result"]
    diff           = out["diff"]
    pixels_changed = out["pixels_changed"]

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown(
            '<div class="panel-label">① Binary Input (Stage 2)</div>',
            unsafe_allow_html=True,
        )
        st.image(binary, width="stretch", clamp=True)

    with col2:
        st.markdown(
            f'<div class="panel-label">② After {operation}</div>',
            unsafe_allow_html=True,
        )
        if result is not None:
            st.image(result, width="stretch", clamp=True)
        else:
            st.markdown(
                '<div class="err-box">Morphology operation failed.</div>',
                unsafe_allow_html=True,
            )

    with col3:
        st.markdown(
            '<div class="panel-label">③ Pixel Diff (changed regions)</div>',
            unsafe_allow_html=True,
        )
        if diff is not None:
            diff_display = (diff > 0).astype(np.uint8) * 255
            st.image(diff_display, width="stretch", clamp=True)
            st.markdown(
                f'<div style="font-size:0.72rem;color:#555c78;text-align:center;">'
                f'{pixels_changed:,} px changed</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="err-box">Diff unavailable.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    stats_col, pad_col = st.columns([2, 1], gap="medium")

    with stats_col:
        st.markdown(
            '<div class="panel-label">📊 &nbsp; Morphology Stats</div>',
            unsafe_allow_html=True,
        )

        total_px = binary.size

        def _fg_pct(img: Optional[np.ndarray]) -> str:
            if img is None:
                return "—"
            return f"{round(np.sum(img == 255) / img.size * 100, 1)}%"

        stat_rows = [
            ("Operation",        operation),
            ("Kernel size",      f"{kernel_size}×{kernel_size}"),
            ("Iterations",       str(iterations)),
            ("Foreground — in",  _fg_pct(binary)),
            ("Foreground — out", _fg_pct(result)),
            ("Pixels changed",   f"{pixels_changed:,} / {total_px:,}"),
            ("Change ratio",     f"{round(pixels_changed / total_px * 100, 2)}%"),
        ]

        rows_html = "".join(
            f"""
            <div class="meta-row">
                <span class="meta-key">{k}</span>
                <span class="meta-val">{v}</span>
            </div>
            """
            for k, v in stat_rows
        )
        st.markdown(f'<div class="meta-card">{rows_html}</div>', unsafe_allow_html=True)

    with pad_col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
                ✅ &nbsp; Stage 3 complete.<br>
                Refined binary passed to Stage 4.
            </div>
            """,
            unsafe_allow_html=True,
        )


def _metric_card(label: str, value: str, colour_hex: str = "#4f8ef7") -> str:
    return f"""
    <div style="background:#11141c;border:1px solid #1e2230;border-top:3px solid {colour_hex};
                border-radius:8px;padding:0.8rem 1rem;text-align:center;min-width:110px;">
        <div style="font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;
                    color:#555c78;margin-bottom:0.3rem;">{label}</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
                    font-weight:700;color:#c8ccd8;">{value}</div>
    </div>"""


def _contour_table_html(infos: list[ContourInfo]) -> str:
    header_style = (
        "background:#0d0f14;padding:0.4rem 0.6rem;text-align:left;"
        "font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;"
        "color:#4f8ef7;border-bottom:1px solid #1e2230;"
    )
    cell_style = (
        "padding:0.35rem 0.6rem;font-size:0.78rem;color:#c8ccd8;"
        "border-bottom:1px solid #141720;"
    )
    dot_style = (
        "display:inline-block;width:10px;height:10px;"
        "border-radius:50%;margin-right:6px;vertical-align:middle;"
    )

    headers = ["#", "Colour", "Area (px²)", "Perimeter", "Centroid",
               "BBox (w×h)", "Aspect", "Extent", "Circularity"]

    header_row = "".join(f"<th style='{header_style}'>{h}</th>" for h in headers)

    rows = ""
    for c in infos:
        dot = f"<span style='{dot_style}background:{c.colour_hex};'></span>"
        rows += f"""
        <tr>
            <td style='{cell_style}'>{c.index + 1}</td>
            <td style='{cell_style}'>{dot}{c.colour_hex}</td>
            <td style='{cell_style}'>{c.area:,.0f}</td>
            <td style='{cell_style}'>{c.perimeter:.1f}</td>
            <td style='{cell_style}'>({c.cx}, {c.cy})</td>
            <td style='{cell_style}'>{c.bbox_w}×{c.bbox_h}</td>
            <td style='{cell_style}'>{c.aspect_ratio}</td>
            <td style='{cell_style}'>{c.extent}</td>
            <td style='{cell_style}'>{c.circularity}</td>
        </tr>"""

    return f"""
    <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;background:#0b0d12;
                  border:1px solid #1e2230;border-radius:8px;overflow:hidden;">
        <thead><tr>{header_row}</tr></thead>
        <tbody>{rows}</tbody>
    </table>
    </div>"""


def render_contour_panel(
    rgb_image: np.ndarray,
    morph_result: np.ndarray,
    params: dict,
) -> None:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-label">🔶 &nbsp; Stage 4 — Contour Detection & Measurements</div>',
        unsafe_allow_html=True,
    )

    out = run_contour_analysis(rgb_image, morph_result, **params)

    infos:      list[ContourInfo]    = out["contours"]
    overlay:    Optional[np.ndarray] = out["overlay"]
    count:      int                  = out["count"]
    total_area: float                = out["total_area"]

    if infos:
        largest  = infos[0]
        avg_area = total_area / count if count else 0

        cards_html = "".join([
            _metric_card("Contours found",  str(count)),
            _metric_card("Total area (px²)", f"{total_area:,.0f}"),
            _metric_card("Avg area (px²)",   f"{avg_area:,.0f}"),
            _metric_card("Largest area",      f"{largest.area:,.0f}"),
            _metric_card("Largest perimeter", f"{largest.perimeter:.1f}"),
        ])
        st.markdown(
            f'<div style="display:flex;gap:0.7rem;flex-wrap:wrap;margin-bottom:1.2rem;">'
            f'{cards_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="info-box" style="border-color:#f7a84f;color:#f7c97e;background:#1a1200;">
                ⚠️ &nbsp; No contours detected with current settings.<br>
                Try lowering the <b>Min contour area</b> or adjusting preprocessing parameters.
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_overlay, col_binary = st.columns([3, 2], gap="large")

    with col_overlay:
        st.markdown(
            '<div class="panel-label">① Contour Overlay</div>',
            unsafe_allow_html=True,
        )
        if overlay is not None:
            st.image(overlay, width="stretch")
        else:
            st.markdown(
                '<div class="err-box">Overlay could not be generated.</div>',
                unsafe_allow_html=True,
            )

    with col_binary:
        st.markdown(
            '<div class="panel-label">② Binary Source (Stage 3 Output)</div>',
            unsafe_allow_html=True,
        )
        st.image(morph_result, width="stretch", clamp=True)

    if infos:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-label">③ Per-Contour Measurements</div>',
            unsafe_allow_html=True,
        )
        st.markdown(_contour_table_html(infos), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-label">④ Contour Detail Cards</div>',
            unsafe_allow_html=True,
        )

        cols_per_row = 4
        for row_start in range(0, len(infos), cols_per_row):
            row_infos = infos[row_start: row_start + cols_per_row]
            cols = st.columns(len(row_infos), gap="small")
            for col, c in zip(cols, row_infos):
                with col:
                    st.markdown(
                        f"""
                        <div style="background:#11141c;border:1px solid #1e2230;
                                    border-left:4px solid {c.colour_hex};
                                    border-radius:8px;padding:0.8rem 0.9rem;
                                    margin-bottom:0.5rem;">
                            <div style="font-family:'Syne',sans-serif;font-weight:700;
                                        font-size:0.9rem;color:{c.colour_hex};
                                        margin-bottom:0.5rem;">Contour #{c.index + 1}</div>
                            <div style="font-size:0.74rem;color:#7a8097;line-height:1.9;">
                                <b style="color:#c8ccd8;">Area</b><br>
                                {c.area:,.0f} px²<br>
                                <b style="color:#c8ccd8;">Perimeter</b><br>
                                {c.perimeter:.1f} px<br>
                                <b style="color:#c8ccd8;">Centroid</b><br>
                                ({c.cx}, {c.cy})<br>
                                <b style="color:#c8ccd8;">BBox</b><br>
                                {c.bbox_w}×{c.bbox_h} at ({c.bbox_x},{c.bbox_y})<br>
                                <b style="color:#c8ccd8;">Circularity</b><br>
                                {c.circularity}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="info-box">
            ✅ &nbsp; Stage 4 complete. &nbsp;
            <span style="color:#c8ccd8;">{count}</span> contour(s) detected and measured.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shape_panel(
    overlay_from_stage4: np.ndarray,
    contour_infos: list[ContourInfo],
    params: dict,
) -> Optional[list[ShapeInfo]]:
    if not params.get("enabled") or not contour_infos:
        return None

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-label">🔺 &nbsp; Stage 5 — Shape Detection</div>',
        unsafe_allow_html=True,
    )

    shape_infos = analyze_shapes(contour_infos)

    labeled_overlay = overlay_from_stage4
    if params.get("show_labels") and overlay_from_stage4 is not None:
        labeled_overlay = draw_shape_labels(overlay_from_stage4, contour_infos, shape_infos)

    col_img, col_table = st.columns([3, 2], gap="large")

    with col_img:
        st.markdown('<div class="panel-label">① Shape-Labeled Overlay</div>', unsafe_allow_html=True)
        if labeled_overlay is not None:
            st.image(labeled_overlay, width="stretch")

    with col_table:
        st.markdown('<div class="panel-label">② Shape Classification Results</div>', unsafe_allow_html=True)

        from collections import Counter
        shape_counts = Counter(s.shape_name for s in shape_infos)

        chips_html = "".join(
            f'<span style="display:inline-block;background:#0e1e3d;border:1px solid #4f8ef7;'
            f'border-radius:20px;padding:0.15rem 0.6rem;font-size:0.72rem;'
            f'color:#7eb8f7;margin:0.15rem;">{name} × {cnt}</span>'
            for name, cnt in shape_counts.most_common()
        )
        st.markdown(
            f'<div style="margin-bottom:0.8rem;">{chips_html}</div>',
            unsafe_allow_html=True,
        )

        th_s = ("background:#0d0f14;padding:0.35rem 0.5rem;text-align:left;"
                "font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;"
                "color:#4f8ef7;border-bottom:1px solid #1e2230;")
        td_s = "padding:0.3rem 0.5rem;font-size:0.75rem;color:#c8ccd8;border-bottom:1px solid #141720;"

        headers = ["#", "Shape", "Vertices", "Confidence", "Area (px²)"]
        hrow = "".join(f"<th style='{th_s}'>{h}</th>" for h in headers)

        rows = ""
        for s in shape_infos:
            ci = next((c for c in contour_infos if c.index == s.contour_index), None)
            dot = f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{ci.colour_hex if ci else "#888"};margin-right:5px;vertical-align:middle;"></span>'
            conf_bar_w = int(s.confidence * 60)
            conf_html = (
                f'<div style="display:flex;align-items:center;gap:0.4rem;">'
                f'<div style="width:{conf_bar_w}px;height:6px;background:#4f8ef7;'
                f'border-radius:3px;"></div>'
                f'<span style="font-size:0.72rem;color:#7a8097;">{s.confidence:.0%}</span>'
                f'</div>'
            )
            rows += (
                f"<tr>"
                f"<td style='{td_s}'>{dot}{s.contour_index + 1}</td>"
                f"<td style='{td_s}'><b>{s.shape_name}</b></td>"
                f"<td style='{td_s}'>{s.vertices_count}</td>"
                f"<td style='{td_s}'>{conf_html}</td>"
                f"<td style='{td_s}'>{s.contour_area:,.0f}</td>"
                f"</tr>"
            )

        st.markdown(
            f'<div style="overflow-x:auto;">'
            f'<table style="width:100%;border-collapse:collapse;background:#0b0d12;'
            f'border:1px solid #1e2230;border-radius:8px;">'
            f'<thead><tr>{hrow}</tr></thead><tbody>{rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="info-box" style="margin-top:0.8rem;">✅ &nbsp; Stage 5 complete. &nbsp;'
        f'<span style="color:#c8ccd8;">{len(shape_infos)}</span> shape(s) classified.</div>',
        unsafe_allow_html=True,
    )
    return shape_infos


def render_skeleton_panel(
    rgb_image: np.ndarray,
    binary: np.ndarray,
    params: dict,
) -> None:
    if not params.get("enabled"):
        return

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-label">🦴 &nbsp; Stage 6 — Skeletonization & Thinning</div>',
        unsafe_allow_html=True,
    )

    if not _SKIMAGE_AVAILABLE:
        st.markdown(
            '<div class="err-box">scikit-image is not installed. '
            'Run <code>pip install scikit-image</code> and restart the app.</div>',
            unsafe_allow_html=True,
        )
        return

    mode      = params["mode"]
    thickness = params["overlay_thickness"]
    mode_label = "Skeleton (medial axis)" if mode == "skeleton" else "Thinning (Guo-Hall)"

    out = run_skeleton_pipeline(rgb_image, binary, mode=mode, overlay_thickness=thickness)

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown('<div class="panel-label">① Binary Input (Stage 3)</div>', unsafe_allow_html=True)
        st.image(binary, width="stretch", clamp=True)

    with col2:
        st.markdown(f'<div class="panel-label">② {mode_label}</div>', unsafe_allow_html=True)
        if out["result"] is not None:
            st.image(out["result"], width="stretch", clamp=True)
        else:
            st.markdown('<div class="err-box">Skeletonization failed.</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="panel-label">③ Skeleton Overlay</div>', unsafe_allow_html=True)
        if out["overlay"] is not None:
            st.image(out["overlay"], width="stretch")
        else:
            st.markdown('<div class="err-box">Overlay unavailable.</div>', unsafe_allow_html=True)

    stat_rows = [
        ("Mode",            mode_label),
        ("Binary pixels",   f"{out['binary_pixels']:,}"),
        ("Skeleton pixels", f"{out['skel_pixels']:,}"),
        ("Reduction",       f"{out['reduction_pct']}%"),
    ]
    rows_html = "".join(
        f'<div class="meta-row"><span class="meta-key">{k}</span>'
        f'<span class="meta-val">{v}</span></div>'
        for k, v in stat_rows
    )
    st.markdown(f'<div class="meta-card" style="max-width:360px;">{rows_html}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box" style="margin-top:0.8rem;">✅ &nbsp; Stage 6 complete.</div>',
        unsafe_allow_html=True,
    )


def render_3d_panel(
    contour_infos: list[ContourInfo],
    image_height: int,
    params: dict,
) -> Optional[np.ndarray]:
    if not params.get("enabled") or not contour_infos:
        return None

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-label">🧊 &nbsp; Stage 7 — 3D Visualisation</div>',
        unsafe_allow_html=True,
    )

    try:
        import plotly.graph_objects as go
        _plotly_ok = True
    except ImportError:
        _plotly_ok = False

    if not _plotly_ok:
        st.markdown(
            '<div class="err-box">plotly not installed. Run: <code>pip install plotly</code></div>',
            unsafe_allow_html=True,
        )
        return None

    extrude    = params["extrude"]
    depth      = params["depth"]
    mode       = params["render_mode"]
    point_size = params["point_size"]

    fig = render_3d_figure(
        contour_infos,
        image_height=image_height,
        extrude=extrude,
        depth=depth,
        point_size=point_size,
        render_mode=mode,
    )

    if fig:
        st.plotly_chart(fig, width="stretch")

    points = (build_extruded_pointcloud(contour_infos, depth=depth, image_height=image_height)
              if extrude else
              build_pointcloud_from_contours(contour_infos, image_height=image_height))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Point cloud size", f"{len(points):,} pts")
    col_b.metric("Contours rendered", str(len(contour_infos)))
    col_c.metric("Mode", mode.title())

    st.markdown(
        '<div class="info-box">✅ &nbsp; Stage 7 complete. Rotate/zoom the chart freely.</div>',
        unsafe_allow_html=True,
    )
    return points


def render_transform_panel(
    contour_infos: list[ContourInfo],
    overlay: Optional[np.ndarray],
    params: dict,
) -> None:
    if not params.get("enabled") or not contour_infos:
        return

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-label">🔄 &nbsp; Stage 8 — Geometric Transformations</div>',
        unsafe_allow_html=True,
    )

    from modules.contour_analysis import draw_contours_overlay

    transform_kwargs = {k: v for k, v in params.items() if k != "enabled"}
    transformed_infos = []
    for ci in contour_infos:
        pts = ci.points.reshape(-1, 2).astype(np.float64)
        t_pts = apply_2d_transforms(pts, **transform_kwargs)
        import copy
        new_ci = copy.copy(ci)
        new_ci.points = t_pts.astype(np.int32).reshape(-1, 1, 2)
        transformed_infos.append(new_ci)

    h, w = (overlay.shape[:2] if overlay is not None else (512, 512))
    blank = np.full((h, w, 3), 20, dtype=np.uint8)

    before_img = draw_contours_overlay(blank, contour_infos,
                                       show_bbox=False, show_centroid=False, show_labels=True)
    after_img  = draw_contours_overlay(blank, transformed_infos,
                                       show_bbox=False, show_centroid=False, show_labels=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="panel-label">Before transformation</div>', unsafe_allow_html=True)
        st.image(before_img, width="stretch")
    with col2:
        st.markdown('<div class="panel-label">After transformation</div>', unsafe_allow_html=True)
        st.image(after_img, width="stretch")

    active = [k for k, v in [
        ("Rotate", params["angle_deg"] != 0),
        ("Scale",  params["scale"]     != 1.0),
        ("Translate", params["tx"] != 0 or params["ty"] != 0),
        ("Reflect", params["reflect_axis"] != "none"),
        ("Shear",  params["shx"] != 0),
    ] if v]
    st.markdown(
        f'<div class="info-box">✅ &nbsp; Stage 8 complete. '
        f'Active: <b>{", ".join(active) or "none"}</b></div>',
        unsafe_allow_html=True,
    )


def render_export_panel(
    images: dict[str, Optional[np.ndarray]],
    points: Optional[np.ndarray],
) -> None:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-label">💾 &nbsp; Stage 9 — Export Results</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(images) + (1 if points is not None else 0), gap="small")
    col_idx = 0

    for label, img in images.items():
        if img is None:
            col_idx += 1
            continue
        png_bytes = image_to_png_bytes(img)
        if png_bytes:
            with cols[col_idx]:
                st.download_button(
                    label=f"⬇ {label}.png",
                    data=png_bytes,
                    file_name=f"{label.lower().replace(' ', '_')}.png",
                    mime="image/png",
                    width="stretch",
                )
        col_idx += 1

    if points is not None and len(points) > 0:
        ply_bytes = pointcloud_to_ply_bytes(points)
        if ply_bytes:
            with cols[col_idx]:
                st.download_button(
                    label="⬇ pointcloud.ply",
                    data=ply_bytes,
                    file_name="pointcloud.ply",
                    mime="application/octet-stream",
                    width="stretch",
                )

    st.markdown(
        '<div class="info-box" style="margin-top:0.6rem;">✅ &nbsp; Click any button to download.</div>',
        unsafe_allow_html=True,
    )


def render_placeholder() -> None:
    st.markdown(
        """
        <div class="placeholder-box">
            <div style="font-size:2.5rem;margin-bottom:0.8rem;">🖼️</div>
            <div style="color:#555c78;font-size:0.9rem;">
                No image uploaded yet.<br>
                Use the uploader above to begin.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    render_sidebar()
    render_hero()

    st.markdown('<div class="panel-label">⬆ &nbsp; Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Drop an image or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        help="Supported: JPG, PNG, BMP, TIFF, WEBP",
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    if uploaded_file is not None:
        logger.info("File received: name=%s size=%s bytes", uploaded_file.name, uploaded_file.size)

        raw_bytes = uploaded_file.read()
        image = load_image_from_bytes(raw_bytes)

        if image is None:
            st.markdown(
                """
                <div class="err-box">
                    ⚠️ &nbsp; Could not decode the uploaded file.<br>
                    Please upload a valid image (JPG, PNG, BMP, TIFF, or WEBP).
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            metadata = get_image_metadata(image)
            render_image_panel(image, metadata)

            params = st.session_state.get("preproc_params", {})
            if params:
                render_preprocessing_panel(image, params)

                morph_params = st.session_state.get("morph_params", {})
                preproc_results = run_preprocessing_pipeline(image, **params)
                binary = preproc_results.get("binary")
                if morph_params and binary is not None:
                    render_morphology_panel(binary, morph_params)

                    contour_params = st.session_state.get("contour_params", {})
                    morph_out = run_morphology(binary, **morph_params)
                    refined_binary = morph_out.get("result") if morph_out.get("result") is not None else binary
                    if contour_params:
                        render_contour_panel(image, refined_binary, contour_params)

                        stage4_out = run_contour_analysis(image, refined_binary, **contour_params)
                        contour_infos = stage4_out["contours"]
                        stage4_overlay = stage4_out["overlay"]

                        shape_params = st.session_state.get("shape_params", {})
                        if shape_params and contour_infos:
                            render_shape_panel(stage4_overlay, contour_infos, shape_params)

                        skel_params = st.session_state.get("skeleton_params", {})
                        skel_result = None
                        if skel_params:
                            render_skeleton_panel(image, refined_binary, skel_params)
                            if skel_params.get("enabled") and _SKIMAGE_AVAILABLE:
                                from modules.skeleton import run_skeleton_pipeline as _rsp
                                _sk_out = _rsp(image, refined_binary, **{k: v for k, v in skel_params.items() if k != "enabled"})
                                skel_result = _sk_out.get("result")

                        mesh_params = st.session_state.get("mesh_params", {})
                        point_cloud = None
                        if mesh_params and contour_infos:
                            point_cloud = render_3d_panel(
                                contour_infos, image.shape[0], mesh_params
                            )

                        transform_params = st.session_state.get("transform_params", {})
                        if transform_params and contour_infos:
                            render_transform_panel(contour_infos, stage4_overlay, transform_params)

                        export_images = {
                            "Original":    image,
                            "Contour Overlay": stage4_overlay,
                            "Skeleton":    skel_result,
                        }
                        render_export_panel(export_images, point_cloud)
    else:
        render_placeholder()


if __name__ == "__main__":
    main()