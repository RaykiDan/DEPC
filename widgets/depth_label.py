from __future__ import annotations

import numpy as np

from PyQt5.QtWidgets import QLabel, QToolTip, QSizePolicy
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt

from config import DMIN, DMAX


class DepthLabel(QLabel):
    """
    A QLabel that displays a depth map and shows a depth-value tooltip
    when the mouse hovers over it.

    How to use in Qt Designer
    -------------------------
    1. Place a regular QLabel where you want the depth display.
    2. Right-click it → "Promote to..." → class name: DepthLabel
       → header file: widgets.depth_label
    3. That's it. No runtime swapping needed.

    How to feed depth data
    ----------------------
    After setting a new pixmap each frame, call:

        label.set_depth(depth_array, dmin, dmax, source, is_normalised)

    Parameters
    ----------
    depth_array  : H×W float32 NumPy array.
                   • Intel RealSense → raw metres   (is_normalised=False)
                   • DAv2 relative   → unitless map (is_normalised=True,
                     will be linearly mapped onto [dmin, dmax] for display)
    dmin / dmax  : Depth range used for the colormap ruler (metres).
    source       : Short label shown in tooltip, e.g. "Intel" or "DAv2".
    is_normalised: See above.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Size policy: never grow beyond what Qt Designer assigned ──────
        #   This is what prevents the depth frames from stealing vertical
        #   space from the RGB row above them.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        # ── Enable hover without needing a mouse button held down ─────────
        self.setMouseTracking(True)

        # ── Alignment: centre the pixmap inside the label ─────────────────
        self.setAlignment(Qt.AlignCenter)

        # ── Internal state ────────────────────────────────────────────────
        self._depth      : np.ndarray | None = None
        self._dmin       : float = DMIN
        self._dmax       : float = DMAX
        self._source     : str   = ""
        self._is_norm    : bool  = False

    # ── Public API ────────────────────────────────────────────────────────────

    def set_depth(
        self,
        depth_array  : np.ndarray | None,
        dmin         : float = DMIN,
        dmax         : float = DMAX,
        source       : str   = "",
        is_normalised: bool  = False,
    ):
        """
        Attach a depth map to this label for hover lookup.
        Pass None to clear (tooltip will be hidden on hover).
        """
        self._depth   = depth_array
        self._dmin    = dmin
        self._dmax    = dmax
        self._source  = source
        self._is_norm = is_normalised

    def clear_depth(self):
        """Remove the depth map — hover tooltip will show nothing."""
        self._depth = None

    # ── Qt events ─────────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

        if self._depth is None:
            QToolTip.hideText()
            return

        pixel = self._label_to_frame(event.x(), event.y())
        if pixel is None:
            QToolTip.hideText()
            return

        fx, fy    = pixel
        raw_value = float(self._depth[fy, fx])

        depth_m = self._to_metres(raw_value)
        tip     = self._format_tip(fx, fy, depth_m)

        QToolTip.showText(QCursor.pos(), tip, self)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        QToolTip.hideText()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _label_to_frame(self, lx: int, ly: int) -> tuple[int, int] | None:
        """
        Map a cursor position inside the QLabel back to the corresponding
        pixel in the original depth array.

        Because we scale pixmaps with Qt.KeepAspectRatio, the rendered image
        may not fill the full label — there can be empty bars on the sides or
        top/bottom. This method accounts for that offset so the tooltip always
        shows the correct pixel's depth.

        Returns (fx, fy) in frame coordinates, or None if the cursor is
        outside the rendered image area.
        """
        if self._depth is None:
            return None

        lw, lh = self.width(), self.height()
        fh, fw = self._depth.shape[:2]

        if lw <= 0 or lh <= 0 or fw <= 0 or fh <= 0:
            return None

        # Scale factor that Qt.KeepAspectRatio would use
        scale = min(lw / fw, lh / fh)

        # Actual rendered image size
        rw = int(fw * scale)
        rh = int(fh * scale)

        # Top-left corner of rendered image inside the label (centred)
        rx0 = (lw - rw) // 2
        ry0 = (lh - rh) // 2

        # Reject if cursor is in the empty bar area
        if not (rx0 <= lx < rx0 + rw and ry0 <= ly < ry0 + rh):
            return None

        # Map back to frame coordinates and clamp to valid range
        fx = int(np.clip((lx - rx0) / scale, 0, fw - 1))
        fy = int(np.clip((ly - ry0) / scale, 0, fh - 1))
        return fx, fy

    def _to_metres(self, raw_value: float) -> float:
        """
        Convert a raw depth value to metres.

        • is_normalised=False → value is already in metres (Intel SDK output).
        • is_normalised=True  → value is in the DAv2 relative range; linearly
          map it onto [dmin, dmax] so the tooltip reads an approximate metre value.
        """
        if self._is_norm:
            return self._dmin + raw_value * (self._dmax - self._dmin)
        return raw_value

    def _format_tip(self, fx: int, fy: int, depth_m: float) -> str:
        """Build the tooltip string shown on hover."""
        prefix = f"[{self._source}] " if self._source else ""

        if depth_m <= 0 or not np.isfinite(depth_m):
            return f"{prefix}({fx}, {fy})  —  No data"

        return f"{prefix}({fx}, {fy})  →  {depth_m:.3f} m"