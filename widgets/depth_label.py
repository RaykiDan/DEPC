from __future__ import annotations

import cv2
import numpy as np

from PyQt5.QtWidgets import QLabel, QToolTip, QSizePolicy
from PyQt5.QtGui import QCursor, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint

from config import DMIN, DMAX


class DepthLabel(QLabel):
    """
    A QLabel that:
      1. Displays a depth map with a colorized overlay.
      2. Shows a depth-value tooltip on hover.
      3. Lets the user draw a Region of Interest (ROI) by click-dragging.
         The average depth within the ROI is overlaid on the frame.

    How to use in Qt Designer
    -------------------------
    Promote any QLabel to DepthLabel with header: widgets.depth_label

    How to feed data each frame
    ---------------------------
        label.update_display(colored_rgb, depth_array)

    This replaces the old setPixmap() + set_depth() pattern.
    The label handles rendering the overlay and pixmap internally.

    ROI controls
    ------------
    - Click and drag on the label to draw an ROI rectangle.
    - The average depth inside the ROI is shown as overlay text.
    - Call clear_roi() to remove the current ROI.
    """

    # ── Overlay appearance ────────────────────────────────────────────────────
    ROI_COLOR       = (0, 255, 0)       # BGR green rectangle
    ROI_THICKNESS   = 2
    TEXT_COLOR      = (0, 255, 0)       # BGR green text
    TEXT_SCALE      = 0.6
    TEXT_THICKNESS  = 2
    TEXT_PADDING    = 6                 # pixels from top-left of ROI box

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Size policy: never expand beyond Qt Designer bounds ───────────
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)

        # ── Depth data ────────────────────────────────────────────────────
        self._depth       : np.ndarray | None = None
        self._colored_rgb : np.ndarray | None = None   # last rendered frame
        self._dmin        : float = DMIN
        self._dmax        : float = DMAX
        self._source      : str   = ""
        self._is_norm     : bool  = False

        # ── ROI state ─────────────────────────────────────────────────────
        #   Stored in FRAME coordinates so the ROI stays correct if the
        #   label is resized. Converted to label coords only for drawing.
        self._roi_start   : tuple[int, int] | None = None   # frame coords
        self._roi_end     : tuple[int, int] | None = None   # frame coords
        self._is_drawing  : bool = False    # True while mouse button is held

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
        Set a new frame for this label.

        Parameters
        ----------
        colored_rgb  : H×W×3 uint8 RGB image (already colormapped).
        depth_array  : H×W float32 depth array (metres or normalised).
        dmin / dmax  : Depth range for tooltip conversion.
        source       : Short label for tooltip, e.g. "Intel" or "DAv2".
        is_normalised: True if depth_array is 0-1 normalised (DAv2 relative).
        """
        self._depth       = depth_array
        self._colored_rgb = colored_rgb.copy()
        self._dmin        = dmin
        self._dmax        = dmax
        self._source      = source
        self._is_norm     = is_normalised

        self._render()

    def clear_roi(self):
        """Remove the current ROI. Called on seek or stop."""
        self._roi_start  = None
        self._roi_end    = None
        self._is_drawing = False
        if self._colored_rgb is not None:
            self._render()

    def clear_depth(self):
        """Remove depth data entirely (called on clear_all)."""
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
        """
        Update depth metadata only (no re-render).
        Used by Intel depth path which sets pixmap separately.
        """
        self._depth   = depth_array
        self._dmin    = dmin
        self._dmax    = dmax
        self._source  = source
        self._is_norm = is_normalised

    # ── Qt mouse events ───────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton and self._depth is not None:
            frame_pos = self._label_to_frame(event.x(), event.y())
            if frame_pos is not None:
                self._roi_start  = frame_pos
                self._roi_end    = frame_pos
                self._is_drawing = True
                self._render()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

        if self._is_drawing and self._depth is not None:
            # ── Update ROI end while dragging ─────────────────────────────
            frame_pos = self._label_to_frame(event.x(), event.y())
            if frame_pos is not None:
                self._roi_end = frame_pos
                self._render()
            QToolTip.hideText()     # suppress hover tooltip while drawing
        else:
            # ── Normal hover tooltip ──────────────────────────────────────
            self._show_hover_tooltip(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and self._is_drawing:
            self._is_drawing = False
            frame_pos = self._label_to_frame(event.x(), event.y())
            if frame_pos is not None:
                self._roi_end = frame_pos
            self._render()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        QToolTip.hideText()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self):
        """
        Compose the final image from the colored frame + ROI overlay,
        then set it as the label pixmap.
        """
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
        """
        Draw the ROI rectangle and average depth text onto the frame.
        Everything is in BGR for OpenCV then converted to RGB at the end.
        """
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        x1, y1 = self._roi_start
        x2, y2 = self._roi_end

        # Normalise so top-left is always the smaller coordinate
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Draw rectangle
        cv2.rectangle(bgr, (x1, y1), (x2, y2), self.ROI_COLOR, self.ROI_THICKNESS)

        # Compute and draw average depth inside ROI
        avg = self._compute_roi_average(x1, y1, x2, y2)
        if avg is not None:
            text = f"avg: {avg:.3f} m"
            tx   = x1 + self.TEXT_PADDING
            ty   = y1 + self.TEXT_PADDING + int(cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self.TEXT_SCALE, self.TEXT_THICKNESS
            )[0][1])

            # Dark background behind text for readability
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self.TEXT_SCALE, self.TEXT_THICKNESS
            )
            cv2.rectangle(
                bgr,
                (tx - 2, ty - th - 2),
                (tx + tw + 2, ty + baseline + 2),
                (0, 0, 0), cv2.FILLED,
            )
            cv2.putText(
                bgr, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, self.TEXT_SCALE,
                self.TEXT_COLOR, self.TEXT_THICKNESS, cv2.LINE_AA,
            )

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── ROI depth calculation ─────────────────────────────────────────────────

    def _compute_roi_average(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> float | None:
        """
        Compute the mean depth value within the ROI rectangle.

        Excludes zero and non-finite values (missing depth data).
        Returns the average in metres, or None if no valid pixels exist
        or the ROI is too small to be meaningful.
        """
        if self._depth is None:
            return None

        # Clamp to frame bounds
        fh, fw = self._depth.shape[:2]
        x1 = max(0, min(x1, fw - 1))
        x2 = max(0, min(x2, fw - 1))
        y1 = max(0, min(y1, fh - 1))
        y2 = max(0, min(y2, fh - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        region = self._depth[y1:y2, x1:x2]
        valid  = region[(region > 0) & np.isfinite(region)]

        if valid.size == 0:
            return None

        raw_avg = float(valid.mean())
        return self._to_metres(raw_avg)

    # ── Coordinate mapping ────────────────────────────────────────────────────

    def _label_to_frame(self, lx: int, ly: int) -> tuple[int, int] | None:
        """
        Map a cursor position inside the QLabel back to the corresponding
        pixel in the original depth/frame array, accounting for the empty
        bars introduced by Qt.KeepAspectRatio scaling.
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
        """Show the per-pixel depth tooltip on hover (when not drawing ROI)."""
        if self._depth is None:
            QToolTip.hideText()
            return

        coords = self._label_to_frame(lx, ly)
        if coords is None:
            QToolTip.hideText()
            return

        fx, fy    = coords
        raw_value = float(self._depth[fy, fx])
        depth_m   = self._to_metres(raw_value)
        tip       = self._format_tip(fx, fy, depth_m)
        QToolTip.showText(QCursor.pos(), tip, self)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_metres(self, raw_value: float) -> float:
        if self._is_norm:
            return self._dmin + raw_value * (self._dmax - self._dmin)
        return raw_value

    def _format_tip(self, fx: int, fy: int, depth_m: float) -> str:
        prefix = f"[{self._source}] " if self._source else ""
        if depth_m <= 0 or not np.isfinite(depth_m):
            return f"{prefix}({fx}, {fy})  —  No data"
        return f"{prefix}({fx}, {fy})  →  {depth_m:.3f} m"