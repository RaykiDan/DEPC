# Depth Estimation Performance on Mobile Robot

A desktop application for comparing monocular depth estimation (Depth Anything V2) against Intel RealSense D435i active-stereo matching depth on a mobile robot platform. Built as part of an undergraduate thesis at Telkom University School of Electrical Engineering.

---

## Screenshot

> *(Insert screenshot of the running application here)*

---

## Features

- **Side-by-side depth visualization** — Depth Anything V2 (DAv2) and Intel RealSense D435i displayed simultaneously with a shared Turbo colormap
- **Metric and relative depth modes** — switch between DAv2 metric (Hypersim, indoor) and relative mode at runtime
- **Encoder selection** — choose between ViT-S, ViT-B, or ViT-L backbone at runtime
- **Interactive hover tooltip** — hover over any depth frame to read the depth value in metres at that pixel
- **Region of Interest (ROI) analysis** — click and drag on any depth frame to draw an ROI; average, min, and max depth are shown in the analysis panel
- **Dataset replay** — load a pre-recorded dataset and replay it with a slider for frame-by-frame navigation
- **Depth ruler** — vertical and horizontal annotated rulers for visual depth reference
- **Configurable depth range** — adjust DMIN/DMAX, encoder, mode, and object annotations from the settings window

---

## Hardware Requirements

- Intel RealSense D435i depth camera
- RGB webcam (tested with Logitech, index configurable)
- PC with CUDA-capable GPU recommended for real-time DAv2 inference
- Raspberry Pi 4 Model B (for mobile robot recording only, not required for replay analysis)

---

## Software Requirements

- Python 3.10+
- PyQt5
- OpenCV (`cv2`)
- PyTorch (CUDA recommended)
- Intel RealSense SDK 2.0 (`pyrealsense2`)
- qfluentwidgets

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pyqt5 opencv-python pyrealsense2 numpy qfluentwidgets
```

---

## Project Structure

```
project/
├── main.py                        # Entry point
├── config.py                      # Shared settings (DMIN, DMAX, encoder, annotations)
├── stream_and_record_V2.py        # Standalone recording program (Program A)
│
├── depth/
│   ├── model.py                   # Depth Anything V2 loader and inference
│   └── realsense.py               # RealSense .bag reader and filter pipeline
│
├── widgets/
│   └── depth_label.py             # Custom QLabel with hover tooltip and ROI drawing
│
├── app/
│   ├── window.py                  # Main application window
│   ├── rulers.py                  # Depth ruler image generators
│   └── setting_window.py          # Settings dialog
│
├── ui/
│   ├── interface.py               # Auto-generated from Qt Designer
│   └── setting.py                 # Auto-generated from Qt Designer
│
├── depth_anything_v2/             # DAv2 model source (metric version)
│   └── dpt.py                     # Must be the metric variant (supports max_depth)
│
├── checkpoints/                   # Model weights (download separately, see below)
│   ├── depth_anything_v2_vits.pth
│   ├── depth_anything_v2_vitb.pth
│   ├── depth_anything_v2_vitl.pth
│   ├── depth_anything_v2_metric_hypersim_vits.pth
│   ├── depth_anything_v2_metric_hypersim_vitb.pth
│   └── depth_anything_v2_metric_hypersim_vitl.pth
│
└── assets/
    └── settings.png               # Settings button icon
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/RaykiDan/DEPC.git
cd DEPC
```

2. Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pyqt5 opencv-python pyrealsense2 numpy qfluentwidgets
```

3. Download model weights (see [Model Weights](#model-weights) section below)

4. Place the metric version of `depth_anything_v2/` at the project root (must support `max_depth` parameter — use the `metric_depth/` variant from the official DAv2 repo)

---

## Usage

### Program A — Recording

Used to capture synchronized data from the Intel RealSense D435i and RGB webcam:

```bash
python stream_and_record_V2.py
```

**Controls:**
- `r` — start/stop recording
- `q` or `ESC` — quit

Each recording session produces:
- `session_YYYYMMDD-HHMMSS.bag` — RealSense raw depth + IR recording
- `ir1_YYYYMMDD-HHMMSS.avi` — IR Left video
- `ir2_YYYYMMDD-HHMMSS.avi` — IR Right video
- `web_YYYYMMDD-HHMMSS.avi` — RGB webcam video
- `depth_YYYYMMDD-HHMMSS.avi` — Depth visualization video (not used in analysis)

Before running, set your webcam index in `stream_and_record_V2.py`:

```python
WEBCAM_INDEX = 8   # change to your webcam index
```

### Program B — Analysis

Used to replay recorded datasets and analyze depth estimation performance:

```bash
python main.py
```

**Steps:**
1. Click **Load** → select a dataset folder containing `cam.avi`, `ir1.avi`, `ir2.avi`, and `recorded.bag`
2. Click **Start** to begin playback
3. Hover over any depth frame to read pixel depth values
4. Click and drag on a depth frame to draw an ROI — statistics appear in the ROI Analysis panel
5. Use the replay slider to navigate frames
6. Click the **Settings** button (gear icon) to change encoder, mode, or depth range

---

## Configuration

All shared settings are in `config.py`:

```python
# Depth display range (metres)
DMIN = 0.2      # closest distance shown on colormap
DMAX = 2.0      # furthest distance shown on colormap

# RealSense threshold filter range (metres)
RS_DMIN = 0.2
RS_DMAX = 3.0

# Active model settings (updated at runtime by Settings window)
CURRENT_ENCODER = "vits"     # "vits" | "vitb" | "vitl"
CURRENT_MODE    = "metric"   # "metric" | "relative"

# Webcam FOV (hardware spec, degrees)
WEBCAM_FOV_H = 64.26
WEBCAM_FOV_V = 50.35

# Horizontal ruler annotations
ANNOTATIONS = [
    {"name": "Kardus",      "depth_min": 1.6, "depth_max": 1.9, "color": (0, 0, 0)},
    {"name": "Kursi",       "depth_min": 1.9, "depth_max": 2.2, "color": (0, 0, 0)},
    {"name": "Papan Tulis", "depth_min": 2.2, "depth_max": 2.7, "color": (0, 0, 0)},
]
```

---

## Model Weights

Download from the official [Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2):

| Model | Mode | Filename |
|---|---|---|
| ViT-S | Relative | `depth_anything_v2_vits.pth` |
| ViT-B | Relative | `depth_anything_v2_vitb.pth` |
| ViT-L | Relative | `depth_anything_v2_vitl.pth` |
| ViT-S | Metric (Hypersim) | `depth_anything_v2_metric_hypersim_vits.pth` |
| ViT-B | Metric (Hypersim) | `depth_anything_v2_metric_hypersim_vitb.pth` |
| ViT-L | Metric (Hypersim) | `depth_anything_v2_metric_hypersim_vitl.pth` |

Place all downloaded `.pth` files in the `checkpoints/` folder.

> **Note:** For metric mode, you must use the `depth_anything_v2/` source from the `metric_depth/` branch of the official repo. The standard relative version does not support the `max_depth` parameter required by metric weights.

---

## Dataset Folder Structure

The analysis program expects each dataset folder to contain exactly:

```
dataset_folder/
├── cam.avi           # RGB webcam recording
├── ir1.avi           # IR Left recording
├── ir2.avi           # IR Right recording
└── recorded.bag      # RealSense raw depth recording
```

These are produced automatically by `stream_and_record_V2.py`. Note: `depth.avi` is also produced during recording but is not used by the analysis program.

---

## Acknowledgements

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — Yang et al., 2024
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [qfluentwidgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)

---

## Reference

Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., & Zhao, H. (2024).
*Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data.*
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).