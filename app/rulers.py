import cv2
import numpy as np

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


# ── Shared colormap helper ────────────────────────────────────────────────────

def _numpy_to_pixmap(rgb_array: np.ndarray) -> QPixmap:
    """Convert an H×W×3 uint8 RGB NumPy array to a QPixmap."""
    h, w, ch = rgb_array.shape
    qimg = QImage(rgb_array.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def depth_to_colormap(depth_m: np.ndarray, dmin: float = 0.5, dmax: float = 4.0) -> np.ndarray:
    """
    Convert a float32 depth array (in metres) to an H×W×3 uint8 RGB image
    using the Turbo colormap.

    Values below dmin → blue end of colormap (close).
    Values above dmax → red end of colormap (far).

    This is the single shared colormap function used by both the depth
    display frames and the ruler — so they always match visually.
    """
    depth_norm  = np.clip((depth_m - dmin) / (dmax - dmin), 0.0, 1.0)
    depth_8u    = (depth_norm * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return colored_rgb


# ── Vertical ruler ────────────────────────────────────────────────────────────

def make_vertical_ruler(
    width : int   = 40,
    height: int   = 250,
    dmin  : float = 0.5,
    dmax  : float = 4.0,
    step  : float = 0.5,
) -> QPixmap:
    """
    Generate a vertical depth scale ruler as a QPixmap.

    The ruler runs top (far = dmax, red) to bottom (close = dmin, blue),
    matching the Turbo colormap convention used in the depth frames.

    Parameters
    ----------
    width, height : Pixel dimensions of the output image.
    dmin, dmax    : Depth range in metres.
    step          : Interval between tick marks in metres.

    Returns
    -------
    QPixmap ready to be passed directly to label.setPixmap().
    """
    ruler = np.zeros((height, width, 3), dtype=np.uint8)

    # ── Draw the color gradient (top = far/red, bottom = close/blue) ──────
    gradient    = np.linspace(1.0, 0.0, height, dtype=np.float32)
    gradient_8u = (gradient * 255).astype(np.uint8)
    turbo       = cv2.applyColorMap(gradient_8u, cv2.COLORMAP_TURBO)

    for y in range(height):
        ruler[y, :] = turbo[y]

    # ── Draw tick marks and labels ────────────────────────────────────────
    tick_w      = max(1, min(5, int(width * 0.3)))
    font_scale  = max(0.3, min(0.46, height / 500))

    ticks = np.arange(dmin, dmax + 1e-6, step)

    for value in ticks:
        rel = (value - dmin) / (dmax - dmin)
        y   = int(np.clip(round((height - 1) * (1.0 - rel)), 0, height - 1))

        # Tick line
        cv2.line(ruler, (0, y), (tick_w, y), (0, 0, 0), 1)

        # Label — whole numbers as "2m", halves as "1.5m"
        label_text = f"{int(round(value))}m" if abs(value - round(value)) < 1e-6 \
                     else f"{value:.1f}m"
        font_size  = 0.40 if abs(value - round(value)) < 1e-6 else 0.36

        cv2.putText(
            ruler, label_text,
            (10, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_size,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

    ruler_rgb = cv2.cvtColor(ruler, cv2.COLOR_BGR2RGB)
    return _numpy_to_pixmap(ruler_rgb)


# ── Horizontal ruler ──────────────────────────────────────────────────────────

def make_horizontal_ruler(
    width      : int   = 1200,
    height     : int   = 40,
    dmin       : float = 0.5,
    dmax       : float = 4.0,
    annotations: list  = None,
) -> QPixmap:
    """
    Generate a horizontal depth scale ruler with optional object annotations.

    The ruler runs left (close = dmin, blue) to right (far = dmax, red).
    Each annotation draws two vertical bracket lines and a centred label
    showing where a known object falls in the depth range.

    Parameters
    ----------
    width, height : Pixel dimensions of the output image.
    dmin, dmax    : Depth range in metres.
    annotations   : List of dicts, each with keys:
                      "name"      : str   — label text
                      "depth_min" : float — near edge of the object (metres)
                      "depth_max" : float — far edge of the object (metres)
                      "color"     : (B, G, R) tuple — text color

    Returns
    -------
    QPixmap ready to be passed directly to label.setPixmap().
    """
    # ── Base gradient (left = close/blue, right = far/red) ────────────────
    grad      = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    ruler_bgr = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
    ruler     = ruler_bgr.copy()

    # ── Draw annotations ──────────────────────────────────────────────────
    if annotations:
        font           = cv2.FONT_HERSHEY_SIMPLEX
        font_scale     = 0.8
        font_thickness = 1

        for obj in annotations:
            d_near = obj.get("depth_min", 0)
            d_far  = obj.get("depth_max", 0)
            name   = obj.get("name",  "Object")
            color  = obj.get("color", (255, 255, 255))

            # Convert depth values to x pixel positions
            x_near = int(np.clip(width * (d_near - dmin) / (dmax - dmin), 0, width - 1))
            x_far  = int(np.clip(width * (d_far  - dmin) / (dmax - dmin), 0, width - 1))

            # Bracket lines at near and far edges
            cv2.line(ruler, (x_near, 0), (x_near, height), (0, 0, 0), 2)
            cv2.line(ruler, (x_far,  0), (x_far,  height), (0, 0, 0), 2)

            # Centred label text
            text_w  = cv2.getTextSize(name, font, font_scale, font_thickness)[0][0]
            text_x  = int(np.clip((x_near + x_far) // 2 - text_w // 2, 5, width - text_w - 5))
            text_y  = int(height * 0.7)

            cv2.putText(
                ruler, name,
                (text_x, text_y),
                font, font_scale,
                color, font_thickness, cv2.LINE_AA,
            )

    ruler_rgb = cv2.cvtColor(ruler, cv2.COLOR_BGR2RGB)
    return _numpy_to_pixmap(ruler_rgb)