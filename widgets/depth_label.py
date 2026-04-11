from __future__ import annotations

import cv2
import numpy as np

from PyQt5.QtWidgets import QLabel, QToolTip, QSizePolicy
from PyQt5.QtGui import QCursor, QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal

from config import DMIN, DMAX


class DepthLabel(QLabel):
    """
    A QLabel that:
      1. Displays a colourised depth map.
      2. Shows a per-pixel depth tooltip on hover.
      3. Lets the user draw a Region of Interest (ROI) by click-dragging.
         The ROI rectangle is drawn on the frame, and stats are emitted
         via the roi_stats_changed signal so the ROI panel can display them.

    How to use in Qt Designer
    -------------------------
    Promote any QLabel to DepthLabel with header: widgets.depth_label

    How to feed data each frame
    ---------------------------
        label.update_display(colored_rgb, depth_array)

    ROI controls
    ------------
    - Click and drag to draw an ROI rectangle.
    - Stats are emitted automatically when the ROI is finalized on release.
    - Call clear_roi() to remove the current ROI.
    """

    # ── Signals ───────────────────────────────────────────────────────────────
    #
    #   Emitted when the user finishes drawing or clears an ROI.
    #   Carries a dict with keys:
    #     "source"  : str          — e.g. "DAv2" or "Intel"
    #     "size"    : (int, int)   — (width, height) of ROI in pixels
    #     "average" : float | None — mean depth in metres
    #     "min"     : float | None — minimum depth in metres
    #     "max"     : float | None — maximum depth in metres
    #   All depth values are None when the ROI is cleared or has no valid pixels.
    #
    roi_stats_changed = pyqtSignal(dict)

    # ── Overlay appearance ────────────────────────────────────────────────────
    ROI_COLOR      = (0, 255, 0)    # BGR green
    ROI_THICKNESS  = 2

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)

        # ── Depth data ────────────────────────────────────────────────────
        self._depth       : np.ndarray | None = None
        self._colored_rgb : np.ndarray | None = None
        self._dmin        : float = DMIN
        self._dmax        : float = DMAX
        self._source      : str   = ""
        self._is_norm     : bool  = False

        # ── ROI state (stored in frame coords) ────────────────────────────
        self._roi_start  : tuple[int, int] | None = None
        self._roi_end    : tuple[int, int] | None = None
        self._is_drawing : bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def update_display(
        self,
        colored_rgb  : np.ndarray,
        depth_array  : np.ndarray,
        dmin         : float = DMIN,
        dmax         : float = DMAX,
        source       : str   = "",
        is_normalised: bool  = False,
    ):
        """
        Set a new frame. Stores both the colourised image and raw depth,
        then re-renders with the current ROI overlay on top.

        Parameters
        ----------
        colored_rgb   : H×W×3 uint8 RGB image (already colormapped).
        depth_array   : H×W float32 array (metres or normalised).
        dmin / dmax   : Depth range for tooltip and ROI stat conversion.
        source        : Short label for tooltip/stats, e.g. "Intel" or "DAv2".
        is_normalised : True if depth_array is 0–1 (DAv2 relative mode).
        """
        self._colored_rgb = colored_rgb.copy()
        self._depth       = depth_array
        self._dmin        = dmin
        self._dmax        = dmax
        self._source      = source
        self._is_norm     = is_normalised

        self._render()

        # Re-emit stats on every new frame so the ROI panel stays current
        # while playback is running with a drawn ROI.
        if self._roi_start is not None and self._roi_end is not None:
            self._emit_stats()

    def clear_roi(self):
        """Remove the current ROI and emit empty stats to clear the panel."""
        self._roi_start  = None
        self._roi_end    = None
        self._is_drawing = False

        self.roi_stats_changed.emit({
            "source":  self._source,
            "size":    None,
            "average": None,
            "min":     None,
            "max":     None,
        })

        if self._colored_rgb is not None:
            self._render()

    def clear_depth(self):
        """Remove all data (called on clear_all)."""
        self._depth       = None
        self._colored_rgb = None
        self._roi_start   = None
        self._roi_end     = None
        self._is_drawing  = False
        self.clear()

    def set_depth(
        self,
        depth_array  : np.ndarray | None,
        dmin         : float = DMIN,
        dmax         : float = DMAX,
        source       : str   = "",
        is_normalised: bool  = False,
    ):
        """Update depth metadata only, without re-rendering."""
        self._depth   = depth_array
        self._dmin    = dmin
        self._dmax    = dmax
        self._source  = source
        self._is_norm = is_normalised

    # ── Qt mouse events ───────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton and self._depth is not None:
            pos = self._label_to_frame(event.x(), event.y())
            if pos is not None:
                self._roi_start  = pos
                self._roi_end    = pos
                self._is_drawing = True
                self._render()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._is_drawing and self._depth is not None:
            pos = self._label_to_frame(event.x(), event.y())
            if pos is not None:
                self._roi_end = pos
                self._render()
            QToolTip.hideText()
        else:
            self._show_hover_tooltip(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and self._is_drawing:
            self._is_drawing = False
            pos = self._label_to_frame(event.x(), event.y())
            if pos is not None:
                self._roi_end = pos
            self._render()
            self._emit_stats()      # ← final stats emitted on release

    def leaveEvent(self, event):
        super().leaveEvent(event)
        QToolTip.hideText()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self):
        """Compose coloured frame + ROI overlay and set as pixmap."""
        if self._colored_rgb is None:
            return

        frame = self._colored_rgb.copy()

        if self._roi_start is not None and self._roi_end is not None:
            frame = self._draw_roi_overlay(frame)

        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.setPixmap(pix)

    def _draw_roi_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw the ROI rectangle onto the frame. No text — stats go to the panel."""
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        x1, y1 = self._roi_start
        x2, y2 = self._roi_end
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        cv2.rectangle(bgr, (x1, y1), (x2, y2), self.ROI_COLOR, self.ROI_THICKNESS)

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _emit_stats(self):
        """
        Calculate full ROI stats and emit roi_stats_changed.
        Called on mouseRelease and on every update_display while ROI exists.
        """
        if self._roi_start is None or self._roi_end is None:
            return

        x1, y1 = self._roi_start
        x2, y2 = self._roi_end
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        region  = self._valid_region(x1, y1, x2, y2)
        w       = x2 - x1
        h       = y2 - y1

        if region is None or region.size == 0:
            self.roi_stats_changed.emit({
                "source":  self._source,
                "size":    (w, h),
                "average": None,
                "min":     None,
                "max":     None,
            })
            return

        avg = self._to_metres(float(region.mean()))
        mn  = self._to_metres(float(region.min()))
        mx  = self._to_metres(float(region.max()))

        self.roi_stats_changed.emit({
            "source":  self._source,
            "size":    (w, h),
            "average": avg,
            "min":     mn,
            "max":     mx,
        })

    def _valid_region(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray | None:
        """
        Return a 1D array of valid (>0, finite) depth values within the ROI.
        Returns None if depth data is missing or the ROI is degenerate.
        """
        if self._depth is None:
            return None

        fh, fw = self._depth.shape[:2]
        x1 = max(0, min(x1, fw - 1))
        x2 = max(0, min(x2, fw - 1))
        y1 = max(0, min(y1, fh - 1))
        y2 = max(0, min(y2, fh - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        region = self._depth[y1:y2, x1:x2]
        valid  = region[(region > 0) & np.isfinite(region)]
        return valid if valid.size > 0 else None

    # ── Coordinate mapping ────────────────────────────────────────────────────

    def _label_to_frame(self, lx: int, ly: int) -> tuple[int, int] | None:
        """
        Map a cursor position inside the QLabel to the corresponding
        pixel in the original depth array, accounting for KeepAspectRatio
        empty bars.
        """
        if self._depth is None:
            return None

        lw, lh = self.width(), self.height()
        fh, fw = self._depth.shape[:2]

        if lw <= 0 or lh <= 0 or fw <= 0 or fh <= 0:
            return None

        scale = min(lw / fw, lh / fh)
        rw    = int(fw * scale)
        rh    = int(fh * scale)
        rx0   = (lw - rw) // 2
        ry0   = (lh - rh) // 2

        if not (rx0 <= lx < rx0 + rw and ry0 <= ly < ry0 + rh):
            return None

        fx = int(np.clip((lx - rx0) / scale, 0, fw - 1))
        fy = int(np.clip((ly - ry0) / scale, 0, fh - 1))
        return fx, fy

    # ── Hover tooltip ─────────────────────────────────────────────────────────

    def _show_hover_tooltip(self, lx: int, ly: int):
        if self._depth is None:
            QToolTip.hideText()
            return

        coords = self._label_to_frame(lx, ly)
        if coords is None:
            QToolTip.hideText()
            return

        fx, fy  = coords
        raw     = float(self._depth[fy, fx])
        depth_m = self._to_metres(raw)
        tip     = self._format_tip(fx, fy, depth_m)
        QToolTip.showText(QCursor.pos(), tip, self)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_metres(self, raw: float) -> float:
        if self._is_norm:
            return self._dmin + raw * (self._dmax - self._dmin)
        return raw

    def _format_tip(self, fx: int, fy: int, depth_m: float) -> str:
        prefix = f"[{self._source}] " if self._source else ""
        if depth_m <= 0 or not np.isfinite(depth_m):
            return f"{prefix}({fx}, {fy})  —  No data"
        return f"{prefix}({fx}, {fy})  →  {depth_m:.3f} m"