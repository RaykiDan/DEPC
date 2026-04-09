import cv2
import numpy as np

from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

from ui.interface import Ui_Form
from qfluentwidgets import setTheme, Theme

from depth.model import DepthModel
from depth.realsense import RealSenseReader
from app.rulers import make_vertical_ruler, make_horizontal_ruler, depth_to_colormap
from config import DMIN, DMAX, WEBCAM_FOV_H, WEBCAM_FOV_V, ANNOTATIONS


class MainApp(QWidget):
    """
    Main application window.

    Responsibilities
    ----------------
    - Owns the UI and connects buttons to actions.
    - Owns a DepthModel and a RealSenseReader instance.
    - Drives the per-frame update loop via QTimer.
    - Converts raw data from the depth modules into QPixmaps for display.

    It does NOT contain any model loading logic, filter chain logic,
    or ruler drawing logic — those all live in their own modules.
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # ── UI setup ──────────────────────────────────────────────────────
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self._apply_stylesheet()

        # ── Depth modules ─────────────────────────────────────────────────
        self.depth_model = DepthModel(encoder="vits", mode="metric")
        self.rs_reader   = RealSenseReader()

        # ── Video captures (RGB and IR feeds from .avi files) ─────────────
        self.cap_cam = None
        self.cap_ir1 = None
        self.cap_ir2 = None

        # ── Playback state ────────────────────────────────────────────────
        self.loaded  = False
        self.playing = False

        # ── Annotations for horizontal ruler ─────────────────────────────
        self.annotations = ANNOTATIONS

        # ── Timer drives the frame loop (~30 fps) ────────────────────────
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frames)

        # ── Connect buttons ───────────────────────────────────────────────
        self.ui.loadButton.clicked.connect(self._select_folder)
        self.ui.startAndStopButton.clicked.connect(self._toggle_play)
        self.ui.clearButton.clicked.connect(self._clear_all)

        # ── Draw initial rulers ───────────────────────────────────────────
        self._update_rulers()

    # ── Qt events ─────────────────────────────────────────────────────────────

    def showEvent(self, event):
        """
        On first show: lock all frame label sizes to their current rendered
        dimensions so they can never expand and push other widgets around.
        """
        super().showEvent(event)

        if getattr(self, "_sizes_locked", False):
            return

        screen    = QApplication.primaryScreen().availableGeometry()
        capped_w  = min(self.maximumWidth(),  screen.width())
        capped_h  = min(self.maximumHeight(), screen.height())
        self.setMaximumSize(capped_w, capped_h)
        self.resize(capped_w, capped_h)
        QApplication.processEvents()

        frame_attrs = (
            "camFrame", "intelLeftFrame", "intelRightFrame",
            "depthFrameCam", "depthFrameIntel", "depthFrameIntel_2",
        )
        for attr in frame_attrs:
            lbl = getattr(self.ui, attr, None)
            if lbl and lbl.width() > 0 and lbl.height() > 0:
                lbl.setMaximumSize(lbl.width(), lbl.height())

        self._sizes_locked = True
        self.resize(self.minimumSize())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_rulers()

    # ── Playback controls ─────────────────────────────────────────────────────

    def _select_folder(self):
        """Open a folder picker and load all sources from the selected dataset."""
        folder = QFileDialog.getExistingDirectory(self, "Pilih folder dataset")
        if not folder:
            return

        # Open .bag for Intel depth
        bag_ok = self.rs_reader.open(f"{folder}/recorded.bag")
        if bag_ok:
            self._update_fov_labels()
            if hasattr(self.ui, "fovCam"):
                self.ui.fovCam.setText(f"FOV: {WEBCAM_FOV_H:.2f}° | V: {WEBCAM_FOV_V:.2f}°")

        # Open RGB and IR video files
        self.cap_cam = self._open_cap(f"{folder}/cam.avi",  "cam.avi")
        self.cap_ir1 = self._open_cap(f"{folder}/ir1.avi",  "ir1.avi")
        self.cap_ir2 = self._open_cap(f"{folder}/ir2.avi",  "ir2.avi")

        # Start playback
        self.loaded  = True
        self.playing = True
        self.timer.start(33)
        self.ui.startAndStopButton.setText("Stop")

    def _toggle_play(self):
        """Pause or resume playback."""
        if not self.loaded:
            print("[WARN] No dataset loaded. Press Load first.")
            return

        self.playing = not self.playing

        if self.playing:
            self.timer.start(33)
            self.ui.startAndStopButton.setText("Stop")
            print("[INFO] Resumed")
        else:
            self.timer.stop()
            self.ui.startAndStopButton.setText("Start")
            print("[INFO] Paused")

    def _clear_all(self):
        """Stop playback and release all resources."""
        self.timer.stop()
        self.playing = False
        self.loaded  = False

        # Release video captures
        for cap in (self.cap_cam, self.cap_ir1, self.cap_ir2):
            if cap:
                cap.release()
        self.cap_cam = self.cap_ir1 = self.cap_ir2 = None

        # Close RealSense pipeline
        self.rs_reader.close()

        # Clear all display labels
        for attr in ("camFrame", "depthFrameCam",
                     "intelLeftFrame", "intelRightFrame",
                     "depthFrameIntel", "depthFrameIntel_2"):
            lbl = getattr(self.ui, attr, None)
            if lbl:
                lbl.clear()

        # Clear hover depth data from DepthLabels
        for attr in ("depthFrameCam", "depthFrameIntel", "depthFrameIntel_2"):
            lbl = getattr(self.ui, attr, None)
            if lbl and hasattr(lbl, "clear_depth"):
                lbl.clear_depth()

        # Reset info labels
        for attr in ("fovLeft", "fovRight", "fovCam", "alphaLeft", "alphaRight"):
            lbl = getattr(self.ui, attr, None)
            if lbl:
                lbl.setText(attr[:3].upper() + ":")

        self.ui.startAndStopButton.setText("Start")
        print("[INFO] Cleared")

    # ── Frame update loop ─────────────────────────────────────────────────────

    def _update_frames(self):
        """Called every ~33 ms by the QTimer. Updates all six display panels."""
        if not self.playing:
            return

        self._update_rgb_frame(self.cap_cam,  self.ui.camFrame)
        self._update_rgb_frame(self.cap_ir1,  self.ui.intelLeftFrame)
        self._update_rgb_frame(self.cap_ir2,  self.ui.intelRightFrame)
        self._update_dav2_depth()
        self._update_intel_depth()

    def _update_rgb_frame(self, cap, label):
        """
        Read one frame from a VideoCapture and display it on a QLabel.
        Loops back to the start of the video when it ends.
        """
        if cap is None or not cap.isOpened():
            label.setText("No video")
            return

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if not ret:
            label.setText("Cannot read frame")
            return

        self._show_bgr_on_label(frame, label)

    def _update_dav2_depth(self):
        """
        Read the current webcam frame, run DAv2 inference on it,
        and display the colorised depth on depthFrameCam.
        """
        if self.cap_cam is None or not self.cap_cam.isOpened():
            return

        # Re-read the current frame position without advancing the capture
        pos = self.cap_cam.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap_cam.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 1))
        ret, frame = self.cap_cam.read()
        if not ret:
            return

        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_depth = self.depth_model.infer(rgb)

        if raw_depth is None:
            self.ui.depthFrameCam.setText("Depth error")
            return

        colored = depth_to_colormap(raw_depth, DMIN, DMAX)
        self._show_rgb_on_depth_label(colored, raw_depth, self.ui.depthFrameCam,
                                      source="DAv2", is_normalised=False)

    def _update_intel_depth(self):
        """
        Read the next depth frame from the .bag file and display
        it on both Intel depth labels.
        """
        depth_m = self.rs_reader.read()
        if depth_m is None:
            return

        colored = depth_to_colormap(depth_m, DMIN, DMAX)
        pixmap  = self._rgb_array_to_pixmap(colored)

        for attr in ("depthFrameIntel", "depthFrameIntel_2"):
            lbl = getattr(self.ui, attr, None)
            if lbl is None:
                continue

            lbl.setPixmap(pixmap.scaled(
                lbl.width(), lbl.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            ))

            # Update hover data on DepthLabels
            if hasattr(lbl, "set_depth"):
                lbl.set_depth(depth_m, DMIN, DMAX, source="Intel", is_normalised=False)

    # ── Rulers ────────────────────────────────────────────────────────────────

    def _update_rulers(self):
        """Regenerate and display all rulers. Called on init and window resize."""
        for attr in ("depthRulerCam", "depthRulerIntel", "depthRulerIntel_2"):
            lbl = getattr(self.ui, attr, None)
            if lbl:
                pixmap = make_vertical_ruler(
                    width=max(40, lbl.width()),
                    height=max(250, lbl.height()),
                    dmin=DMIN, dmax=DMAX,
                )
                lbl.setPixmap(pixmap.scaled(
                    lbl.width(), lbl.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                ))

        if hasattr(self.ui, "depthObjectRuler"):
            lbl    = self.ui.depthObjectRuler
            pixmap = make_horizontal_ruler(
                width=max(100, lbl.width()),
                height=max(40,  lbl.height()),
                dmin=DMIN, dmax=DMAX,
                annotations=self.annotations,
            )
            lbl.setPixmap(QPixmap(pixmap).scaled(
                lbl.width(), lbl.height(),
                Qt.IgnoreAspectRatio, Qt.SmoothTransformation,
            ))

    # ── FOV labels ────────────────────────────────────────────────────────────

    def _update_fov_labels(self):
        """Pull FOV data from RealSenseReader and update the UI labels."""
        fov = self.rs_reader.get_fov()
        if fov is None:
            return

        if hasattr(self.ui, "fovLeft"):
            h, v = fov["ir1"]
            self.ui.fovLeft.setText(f"FOV: {h:.2f}° | V: {v:.2f}°")

        if hasattr(self.ui, "fovRight"):
            h, v = fov["ir2"]
            self.ui.fovRight.setText(f"FOV: {h:.2f}° | V: {v:.2f}°")

        if hasattr(self.ui, "alphaLeft"):
            self.ui.alphaLeft.setText("α: 25")
        if hasattr(self.ui, "alphaRight"):
            self.ui.alphaRight.setText("α: 25")

    # ── Display helpers ───────────────────────────────────────────────────────

    def _show_bgr_on_label(self, bgr_frame: np.ndarray, label):
        """Convert a BGR OpenCV frame to QPixmap and show it on a plain QLabel."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pix = self._rgb_array_to_pixmap(rgb)
        label.setPixmap(pix.scaled(label.width(), label.height()))

    def _show_rgb_on_depth_label(
        self,
        rgb_array  : np.ndarray,
        depth_array: np.ndarray,
        label,
        source     : str  = "",
        is_normalised: bool = False,
    ):
        """
        Show a colorised depth image on a DepthLabel and attach the raw
        depth array for hover lookups.
        """
        pix = self._rgb_array_to_pixmap(rgb_array)
        label.setPixmap(pix.scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        ))

        if hasattr(label, "set_depth"):
            label.set_depth(depth_array, DMIN, DMAX, source, is_normalised)

    @staticmethod
    def _rgb_array_to_pixmap(rgb_array: np.ndarray) -> QPixmap:
        """Convert an H×W×3 uint8 RGB NumPy array to a QPixmap."""
        h, w, ch = rgb_array.shape
        qimg = QImage(rgb_array.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    @staticmethod
    def _open_cap(path: str, name: str):
        """Open a VideoCapture and log the result."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[WARN] Could not open {name}")
        return cap

    # ── Stylesheet ────────────────────────────────────────────────────────────

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #202020;
                color: #FFFFFF;
            }
            QLabel {
                background-color: #2B2B2B;
                color: #FFFFFF;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                padding: 4px;
            }
            QLabel[objectName*="fov"], QLabel[objectName*="alpha"] {
                background-color: transparent;
                border: none;
                padding: 4px 8px;
                font-size: 9pt;
                color: #CCCCCC;
            }
            QLabel[objectName*="Title"], QLabel[objectName*="Label"] {
                background-color: transparent;
                border: none;
                font-size: 12pt;
                font-weight: bold;
                padding: 4px;
            }
            QFrame {
                background-color: #2B2B2B;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #2B2B2B;
                color: #FFFFFF;
                border: 2px solid #3A3A3A;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #3A3A3A;
                border: 2px solid #0078D4;
            }
            QPushButton:pressed {
                background-color: #0078D4;
                border: 2px solid #005A9E;
            }
            QPushButton[objectName="loadButton"],
            QPushButton[objectName="startAndStopButton"],
            QPushButton[objectName="clearButton"] {
                border: 2px solid #0078D4;
            }
            QPushButton[objectName="loadButton"]:hover,
            QPushButton[objectName="startAndStopButton"]:hover,
            QPushButton[objectName="clearButton"]:hover {
                background-color: #0078D4;
                border: 2px solid #005A9E;
            }
            QComboBox {
                background-color: #2B2B2B;
                color: #FFFFFF;
                border: 2px solid #0078D4;
                border-radius: 6px;
                padding: 5px;
                min-width: 80px;
            }
            QComboBox:hover { border: 2px solid #00B7C3; background-color: #3A3A3A; }
            QComboBox:focus { border: 2px solid #00B7C3; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow { image: url(down_arrow.png); width: 12px; height: 12px; }
            QComboBox QAbstractItemView {
                background-color: #2B2B2B;
                color: #FFFFFF;
                selection-background-color: #0078D4;
                border: 1px solid #0078D4;
            }
            QToolTip {
                background-color: #1A1A2E;
                color: #00D4FF;
                border: 1px solid #0078D4;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 10pt;
                font-weight: bold;
            }
        """)