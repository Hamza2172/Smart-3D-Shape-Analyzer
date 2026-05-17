# Smart 3D Shape Analyzer

A full end-to-end computer vision pipeline built with Streamlit, OpenCV, scikit-image, and Plotly. Upload an image and watch it flow through 9 live stages — from raw pixels to an interactive 3D point cloud.

---

## Stages

| #   | Stage                     | Description                                                            |
| --- | ------------------------- | ---------------------------------------------------------------------- |
| 01  | Image Upload & Display    | Load JPG, PNG, BMP, TIFF, or WEBP images with metadata extraction      |
| 02  | Preprocessing             | Grayscale → Noise Reduction → Gaussian Blur → Thresholding             |
| 03  | Morphological Operations  | Dilation, Erosion, Opening, Closing with pixel-diff visualization      |
| 04  | Contour Detection         | Detect, measure, and colour-code all contours with per-contour metrics |
| 05  | Shape Detection           | Classify contours into Triangle, Circle, Rectangle, Polygon, etc.      |
| 06  | Skeletonization           | Medial axis skeleton or Guo-Hall thinning via scikit-image             |
| 07  | 3D Visualization          | Interactive Plotly point cloud or extruded 3D mesh from contours       |
| 08  | Geometric Transformations | Rotate, scale, translate, reflect, and shear contour geometry          |
| 09  | Export                    | Download processed images as PNG and point clouds as PLY               |

---

## Project Structure

```
smart-3d-shape-analyzer/
│
├── app.py                        # Main Streamlit entry point
|
├── dataset                       # Folder Contains Sample images for testing the pipeline across different shape types and lighting conditions
│
└── modules/
    ├── image_loader.py           # Stage 1 — image ingestion & metadata
    ├── preprocessing.py          # Stage 2 — grayscale, blur, thresholding
    ├── morphology.py             # Stage 3 — morphological operations
    ├── contour_analysis.py       # Stage 4 — contour detection & metrics
    ├── shape_detection.py        # Stage 5 — shape classification
    ├── skeleton.py               # Stage 6 — skeletonization & thinning
    ├── mesh_builder.py           # Stage 7 — 3D point cloud & Plotly figure
    ├── transformations.py        # Stage 8 — 2D/3D geometric transforms
    └── export_utils.py           # Stage 9 — PNG & PLY export
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/smart-3d-shape-analyzer.git
cd smart-3d-shape-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Requirements

```
streamlit
opencv-python
numpy
scikit-image
plotly
open3d          # optional — PLY export fallback is built-in
```

Or install all at once:

```bash
pip install streamlit opencv-python numpy scikit-image plotly open3d
```

> **Note:** `open3d` and `scikit-image` are optional. The app detects their availability at runtime and disables the relevant stages gracefully if they are not installed.

---

## Usage

1. Run `streamlit run app.py`
2. Open the browser at `http://localhost:8501`
3. Upload an image using the uploader (JPG, PNG, BMP, TIFF, WEBP)
4. Adjust any stage parameters from the left sidebar
5. All 9 stages update live as you move the sliders
6. Use the **Export** section at the bottom to download results

---

## Sidebar Controls

| Section          | Controls                                                            |
| ---------------- | ------------------------------------------------------------------- |
| Preprocessing    | Blur kernel, threshold value/mode, invert, denoise kernel           |
| Morphology       | Operation (Dilate/Erode/Open/Close), kernel size, iterations        |
| Contour          | Min area, max contours, thickness, overlay toggles                  |
| Shape Detection  | Enable/disable, show labels                                         |
| Skeletonization  | Enable/disable, mode (skeleton / thinning), overlay thickness       |
| 3D Visualization | Enable/disable, render mode (points/lines/surface), extrusion depth |
| Transformations  | Rotation, scale, translation, reflection axis, shear                |

---

## Supported Image Formats

`JPG` · `PNG` · `BMP` · `TIFF` · `WEBP`

---

## Export Formats

| File                  | Content                           |
| --------------------- | --------------------------------- |
| `original.png`        | Original uploaded image           |
| `contour_overlay.png` | Colour-coded contour annotation   |
| `skeleton.png`        | Skeletonized binary image         |
| `pointcloud.ply`      | 3D point cloud (ASCII PLY format) |

---

## License

MIT License — see `LICENSE` for details.

---

_Built with OpenCV · scikit-image · Plotly · Streamlit_
