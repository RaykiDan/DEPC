import sys
import math
import cv2
import time
import torch
import pyrealsense2 as rs
import numpy as np

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ui.interface import Ui_Form
from depth_anything_v2.dpt import DepthAnythingV2
from qfluentwidgets import setTheme, Theme, setThemeColor


def generate_depth_ruler(width=40, height=250, dmin=0.5, dmax=4.0):
    ruler = np.zeros((height, width, 3), dtype=np.uint8)

    # vertical gradient: index 0 = top (red), index height-1 = bottom (blue)
    gradient = np.linspace(1.0, 0.0, height).astype(np.float32)
    gradient_8u = (gradient * 255).astype(np.uint8)
    turbo = cv2.applyColorMap(gradient_8u, cv2.COLORMAP_TURBO)

    # fill ruler
    for y in range(height):
        ruler[y, :] = turbo[y]

    # ticks ruler
    step = 0.5
    ticks = np.arange(dmin, dmax + 1e-6, step)

    # Scale tick positions and text based on height
    tick_width = min(5, int(width * 0.3))
    font_scale = max(0.3, min(0.46, height / 500))
    text_offset = int(width * 0.375)

    for value in ticks:
        rel = (value - dmin) / (dmax - dmin)
        y = int(round((height - 1) * (1.0 - rel)))
        y = np.clip(y, 0, height - 1)

        cv2.line(ruler, (0, y), (tick_width, y), (0, 0, 0), 1)

        if abs(value - round(value)) < 1e-6:
            cv2.putText(
                ruler,
                f"{int(round(value))}m",
                (10, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
        else:
            cv2.putText(
                ruler,
                f"{value:.1f}m",
                (10, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

    return ruler


def generate_horizontal_ruler_with_annotations(width=1200, height=40, dmin=0.5, dmax=4.0,
                                               object_annotations=None):
    # Create gradient bar using full height
    grad = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    # Don't flip - keep it so blue (0.5m) is on left, red (3.5m) is on right

    ruler_bgr = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
    ruler = ruler_bgr.copy()

    # Add object annotations ON TOP of the color bar if provided
    if object_annotations:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 1

        for obj in object_annotations:
            depth_min = obj.get('depth_min', 0)
            depth_max = obj.get('depth_max', 0)
            name = obj.get('name', 'Object')
            text_color = obj.get('color', (255, 255, 255))

            # Calculate x positions based on depth range
            # No flip - depth increases left to right
            rel_pos_min = (depth_min - dmin) / (dmax - dmin)
            rel_pos_max = (depth_max - dmin) / (dmax - dmin)

            x_min = int(width * rel_pos_min)
            x_max = int(width * rel_pos_max)

            x_min = np.clip(x_min, 0, width - 1)
            x_max = np.clip(x_max, 0, width - 1)

            # Draw tick marks at start and end of range (like vertical ruler)
            tick_length = int(height)
            cv2.line(ruler, (x_min, 0), (x_min, tick_length), (0, 0, 0), 2)
            cv2.line(ruler, (x_max, 0), (x_max, tick_length), (0, 0, 0), 2)

            # Draw text on top of the color bar
            text_x = (x_min + x_max) // 2
            text_size = cv2.getTextSize(name, font, font_scale, font_thickness)[0]
            text_x = text_x - text_size[0] // 2
            text_y = int(height * 0.7)

            # Ensure text stays within bounds
            text_x = np.clip(text_x, 5, width - text_size[0] - 5)

            # Draw text
            cv2.putText(ruler, name, (text_x, text_y), font, font_scale,
                        text_color, font_thickness, cv2.LINE_AA)

    ruler_rgb = cv2.cvtColor(ruler, cv2.COLOR_BGR2RGB)
    return ruler_rgb


class MainApp(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # Apply Fluent styling to the main window
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
            /* Special styling for info labels (FOV, alpha) */
            QLabel[objectName*="fov"], QLabel[objectName*="alpha"] {
                background-color: transparent;
                border: none;
                padding: 4px 8px;
                font-size: 9pt;
                font-weight: normal;
                color: #CCCCCC;
            }
            /* Title labels (Original, Intel Right, etc.) */
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
            /* Special styling for specific buttons */
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
            QComboBox:hover {
                border: 2px solid #00B7C3;
                background-color: #3A3A3A;
            }
            QComboBox:focus {
                border: 2px solid #00B7C3;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #2B2B2B;
                color: #FFFFFF;
                selection-background-color: #0078D4;
                border: 1px solid #0078D4;
            }
        """)

        self.playing = False
        self.loaded = False
        self.caps = {}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {self.device}")

        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
        }

        self.depth_model = None

        self.rs_pipeline = None
        self.rs_align = rs.align(rs.stream.infrared)
        self.rs_config = None
        self.rs_profile = None
        self.colorizer = rs.colorizer()

        # ---------- RealSense Filters ----------
        self.dec_filter = rs.decimation_filter()
        self.spa_filter = rs.spatial_filter()
        self.tmp_filter = rs.temporal_filter()
        self.hole_filter = rs.hole_filling_filter()

        self.to_disparity = rs.disparity_transform(True)
        self.to_depth = rs.disparity_transform(False)

        self.th_filter = rs.threshold_filter()
        self.th_filter.set_option(rs.option.min_distance, 0.2)
        self.th_filter.set_option(rs.option.max_distance, 3.0)
        # ---------- ---------- ----------

        self.ui.loadButton.clicked.connect(self.select_folder)
        self.ui.startAndStopButton.clicked.connect(self.toggle_play)
        self.ui.clearButton.clicked.connect(self.clear_all)

        self.cap_cam = None
        self.cap_ir1 = None
        self.cap_ir2 = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)

        self.update_depth_ruler()

        self.encoder_to_size = {
            "vitl": 252,
            "default": 252
        }

        self.checkpoint_paths = {
            "vits": "checkpoints/depth_anything_v2_vits.pth",
            "vitb": "checkpoints/depth_anything_v2_vitb.pth",
            "vitl": "checkpoints/depth_anything_v2_vitl.pth",
        }

        # Set webcam FOV (from calibration results)
        self.webcam_fov_h = 64.26
        self.webcam_fov_v = 50.35

        # Initialize FOV labels
        if hasattr(self.ui, 'fovCam'):
            self.ui.fovCam.setText("FOV:")
        if hasattr(self.ui, 'fovLeft'):
            self.ui.fovLeft.setText("FOV:")
        if hasattr(self.ui, 'fovRight'):
            self.ui.fovRight.setText("FOV:")

        # Initialize Alpha labels
        if hasattr(self.ui, 'alphaLeft'):
            self.ui.alphaLeft.setText("α:")
        if hasattr(self.ui, 'alphaRight'):
            self.ui.alphaRight.setText("α:")

        # Define object annotations with RANGES (ADJUST THESE AS NEEDED!)
        self.object_annotations = [
            {'depth_min': 0.5, 'depth_max': 1.2, 'name': 'Kursi', 'color': (0, 0, 0)},
            {'depth_min': 1.2, 'depth_max': 1.8, 'name': 'Kardus', 'color': (0, 0, 0)},
            {'depth_min': 1.8, 'depth_max': 2.3, 'name': 'Papan Tulis', 'color': (0, 0, 0)},
            {'depth_min': 2.3, 'depth_max': 2.6, 'name': 'Tool', 'color': (0, 0, 0)},
            {'depth_min': 2.6, 'depth_max': 4.0, 'name': 'Dinding', 'color': (0, 0, 0)},
        ]

        # Update horizontal ruler with annotations
        self.update_horizontal_ruler()

    def set_object_annotations(self, annotations):
        """Update object annotations dynamically"""
        self.object_annotations = annotations
        self.update_horizontal_ruler()
        print(f"[INFO] Updated {len(annotations)} object annotations")

    def resizeEvent(self, event):
        """Handle window resize to update rulers"""
        super().resizeEvent(event)
        self.update_depth_ruler()
        self.update_horizontal_ruler()

    def update_horizontal_ruler(self):
        """Update horizontal ruler with current widget size and annotations"""
        if hasattr(self.ui, 'depthObjectRuler'):
            ruler = generate_horizontal_ruler_with_annotations(
                width=self.ui.depthObjectRuler.width(),
                height=self.ui.depthObjectRuler.height(),
                dmin=0.5,
                dmax=4.0,
                object_annotations=self.object_annotations
            )
            h, w, c = ruler.shape
            qimg = QImage(ruler.data, w, h, 3 * w, QImage.Format_RGB888)
            self.ui.depthObjectRuler.setPixmap(QPixmap.fromImage(qimg))

    def get_fov_from_intrinsics(self, intrinsics):
        """Calculate FOV from camera intrinsics"""
        fov_h = 2 * math.atan(intrinsics.width / (2 * intrinsics.fx)) * (180 / math.pi)
        fov_v = 2 * math.atan(intrinsics.height / (2 * intrinsics.fy)) * (180 / math.pi)
        return fov_h, fov_v

    def update_fov_labels(self):
        """Update FOV and Alpha labels from RealSense streams"""
        if self.rs_profile is None:
            return

        try:
            ir1_profile = self.rs_profile.get_stream(rs.stream.infrared, 1)
            ir2_profile = self.rs_profile.get_stream(rs.stream.infrared, 2)

            ir1_intrinsics = ir1_profile.as_video_stream_profile().get_intrinsics()
            ir2_intrinsics = ir2_profile.as_video_stream_profile().get_intrinsics()

            ir1_fov_h, ir1_fov_v = self.get_fov_from_intrinsics(ir1_intrinsics)
            ir2_fov_h, ir2_fov_v = self.get_fov_from_intrinsics(ir2_intrinsics)

            # Update FOV labels
            if hasattr(self.ui, 'fovLeft'):
                self.ui.fovLeft.setText(f"FOV: {ir1_fov_h:.2f}° | V: {ir1_fov_v:.2f}°")

            if hasattr(self.ui, 'fovRight'):
                self.ui.fovRight.setText(f"FOV: {ir2_fov_h:.2f}° | V: {ir2_fov_v:.2f}°")

            print(f"[INFO] Left IR FOV - H: {ir1_fov_h:.2f}°, V: {ir1_fov_v:.2f}°")
            print(f"[INFO] Right IR FOV - H: {ir2_fov_h:.2f}°, V: {ir2_fov_v:.2f}°")

            # Calculate and update Alpha values (aspect ratio angle)
            alpha_left = 25
            alpha_right = 25

            print(f"[DEBUG] Calculated Alpha Left: {alpha_left:.1f}")
            print(f"[DEBUG] Calculated Alpha Right: {alpha_right:.1f}")
            print(f"[DEBUG] Has alphaLeft attr: {hasattr(self.ui, 'alphaLeft')}")
            print(f"[DEBUG] Has alphaRight attr: {hasattr(self.ui, 'alphaRight')}")

            if hasattr(self.ui, 'alphaLeft'):
                self.ui.alphaLeft.setText(f"α: {alpha_left}")
                print(f"[INFO] Set alphaLeft to: α: {alpha_left}")
            else:
                print("[WARN] alphaLeft label not found in UI")

            if hasattr(self.ui, 'alphaRight'):
                self.ui.alphaRight.setText(f"α: {alpha_right}")
                print(f"[INFO] Set alphaRight to: α: {alpha_right}")
            else:
                print("[WARN] alphaRight label not found in UI")

        except Exception as e:
            print(f"[ERROR] Failed to get FOV/Alpha: {e}")

    def toggle_play(self):
        if not self.loaded:
            print("[WARN] Belum ada dataset. Tekan Select dulu.")
            return

        self.playing = not self.playing

        if self.playing:
            print("[INFO] PLAY")
            self.timer.start(33)
            self.ui.startAndStopButton.setText("Stop")
        else:
            print("[INFO] PAUSE")
            self.timer.stop()
            self.ui.startAndStopButton.setText("Start")

    def clear_all(self):
        print("[INFO] CLEAR ALL")

        self.timer.stop()
        self.playing = False
        self.loaded = False

        if self.cap_cam: self.cap_cam.release()
        if self.cap_ir1: self.cap_ir1.release()
        if self.cap_ir2: self.cap_ir2.release()

        self.cap_cam = None
        self.cap_ir1 = None
        self.cap_ir2 = None

        if self.rs_pipeline:
            try:
                self.rs_pipeline.stop()
            except:
                pass
        self.rs_pipeline = None

        self.ui.camFrame.clear()
        self.ui.depthFrameCam.clear()
        self.ui.intelLeftFrame.clear()
        self.ui.intelRightFrame.clear()
        self.ui.depthFrameIntel.clear()
        self.ui.depthFrameIntel_2.clear()

        # Reset FOV labels
        if hasattr(self.ui, 'fovLeft'):
            self.ui.fovLeft.setText("FOV:")
        if hasattr(self.ui, 'fovRight'):
            self.ui.fovRight.setText("FOV:")
        if hasattr(self.ui, 'fovCam'):
            self.ui.fovCam.setText("FOV:")

        # Reset Alpha labels
        if hasattr(self.ui, 'alphaLeft'):
            self.ui.alphaLeft.setText("α:")
        if hasattr(self.ui, 'alphaRight'):
            self.ui.alphaRight.setText("α:")

        self.ui.startAndStopButton.setText("Start")

    def update_depth_ruler(self):
        """Update vertical depth rulers with current widget sizes"""
        ruler_labels = []
        if hasattr(self.ui, 'depthRulerCam'):
            ruler_labels.append(self.ui.depthRulerCam)
        if hasattr(self.ui, 'depthRulerIntel'):
            ruler_labels.append(self.ui.depthRulerIntel)
        if hasattr(self.ui, 'depthRulerIntel_2'):
            ruler_labels.append(self.ui.depthRulerIntel_2)

        for label in ruler_labels:
            width = max(40, label.width())
            height = max(250, label.height())

            ruler_img = generate_depth_ruler(width, height, dmin=0.5, dmax=4.0)
            ruler_img_rgb = cv2.cvtColor(ruler_img, cv2.COLOR_BGR2RGB)

            h, w, ch = ruler_img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(ruler_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimg)
            label.setPixmap(pixmap.scaled(label.width(), label.height(),
                                          aspectRatioMode=1))

    def ensure_model_loaded(self):
        encoder = self.ui.encoderBox.currentText() if hasattr(self.ui, "encoderBox") else "vitl"
        if encoder == "":
            encoder = "vitl"

        if self.depth_model is None or getattr(self, "_loaded_encoder", None) != encoder:
            ckpt = self.checkpoint_paths.get(encoder, None)
            if ckpt is None:
                print(f"[ERROR] Tidak menemukan checkpoint untuk encoder '{encoder}'")
                return

            print(f"[INFO] Loading DepthAnythingV2 model ({encoder}) ...")

            try:
                config = self.model_configs[encoder]
                model = DepthAnythingV2(**config)

                state_dict = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state_dict)

                model = model.to(self.device).eval()
                self.depth_model = model
                self._loaded_encoder = encoder

                print("[INFO] DepthAnythingV2 Loaded Successfully")

            except Exception as e:
                print("[ERROR] Gagal load DepthAnythingV2:", e)

    def infer_dav2(self, frame):
        self.ensure_model_loaded()
        try:
            raw_depth = self.depth_model.infer_image(frame)
        except Exception as e:
            print("[ERROR] infer_image metric DAV2:", e)
            return None

        return raw_depth

    def estimate_depth_anything(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_depth = self.infer_dav2(rgb)

        if raw_depth is None:
            return np.zeros_like(frame), None

        depth_color = self.apply_global_colormap(raw_depth)
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

        return depth_color, raw_depth

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Pilih folder dataset")
        if folder == "":
            return

        cam_path = f"{folder}/cam.avi"
        ir1_path = f"{folder}/ir1.avi"
        ir2_path = f"{folder}/ir2.avi"
        self.bag_path = f"{folder}/recorded.bag"

        try:
            self.rs_pipeline = rs.pipeline()
            self.rs_config = rs.config()
            self.rs_config.enable_device_from_file(self.bag_path, repeat_playback=True)

            self.rs_profile = self.rs_pipeline.start(self.rs_config)

            playback = self.rs_profile.get_device().as_playback()
            playback.set_real_time(False)
            print("[INFO] Berhasil membuka .bag")

            # Update FOV and Alpha labels
            self.update_fov_labels()

            # Update webcam FOV label
            if hasattr(self.ui, 'fovCam'):
                self.ui.fovCam.setText(f"FOV: {self.webcam_fov_h:.2f}° | V: {self.webcam_fov_v:.2f}°")

        except Exception as e:
            print("[ERROR] Gagal membuka .bag", e)
            self.rs_pipeline = None

        self.cap_cam = cv2.VideoCapture(cam_path)
        self.cap_ir1 = cv2.VideoCapture(ir1_path)
        self.cap_ir2 = cv2.VideoCapture(ir2_path)

        if not (self.cap_cam and self.cap_cam.isOpened()):
            print("[ERROR] cam.avi tidak berhasil dibukaa")
        if not (self.cap_ir1 and self.cap_ir1.isOpened()):
            print("[ERROR] ir1.avi tidak berhasil dibuka")
        if not (self.cap_ir2 and self.cap_ir2.isOpened()):
            print("[ERROR] ir2.avi tidak berhasil dibuka")

        self.timer.start(33)
        self.loaded = True
        self.playing = True
        self.ui.startAndStopButton.setText("Stop")

    def update_frames(self):
        if not self.playing:
            return

        self.update_frame(self.cap_cam, self.ui.camFrame, self.ui.depthFrameCam)
        self.update_frame(self.cap_ir1, self.ui.intelLeftFrame)
        self.update_frame(self.cap_ir2, self.ui.intelRightFrame)
        self.update_depth_bag()

    def update_frame(self, cap, rgb_label, depth_label=None):
        if cap is None or not cap.isOpened():
            rgb_label.setText("Tidak ada video")
            if depth_label is not None:
                depth_label.setText("Tidak ada video")
            return

        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                rgb_label.setText("Tidak bisa membaca frame")
                if depth_label is not None:
                    depth_label.setText("Tidak bisa membaca frame")
                return

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = rgb.strides[0]
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(rgb_label.width(), rgb_label.height())
            rgb_label.setPixmap(pix)
        except Exception as e:
            print("[ERROR] Gagal menampilkan frame", e)
            rgb_label.setText("Gagal menampilkan frame")

        if depth_label is not None:
            try:
                depth_color, raw_depth = self.estimate_depth_anything(rgb)

                h2, w2 = depth_color.shape[:2]
                bytes_per_line2 = depth_color.strides[0]
                qimg2 = QImage(depth_color.data, w2, h2, bytes_per_line2, QImage.Format_RGB888)
                pix2 = QPixmap.fromImage(qimg2).scaled(depth_label.width(), depth_label.height())
                depth_label.setPixmap(pix2)

            except Exception as e:
                print("[ERROR] DepthAnything terganggu atau gagal display", e)
                depth_label.setText("Error depth")

    def update_depth_bag(self):
        if self.rs_pipeline is None:
            return

        try:
            frames = self.rs_pipeline.wait_for_frames(timeout_ms=50)
            if self.rs_align:
                frames = self.rs_align.process(frames)

            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return

            filtered = self.apply_depth_filters(depth_frame)

            depth_raw = np.asanyarray(filtered.get_data()).astype(np.float32) * 0.001
            depth_img = self.apply_global_colormap(depth_raw)

            h, w = depth_img.shape[:2]
            bytes_per_line = depth_img.strides[0]
            qimg = QImage(depth_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(self.ui.depthFrameIntel.width(),
                                                 self.ui.depthFrameIntel.height())
            self.ui.depthFrameIntel.setPixmap(pix)
            self.ui.depthFrameIntel_2.setPixmap(pix)

        except Exception as e:
            print("[ERROR] Gagal membaca .bag:", e)

    def apply_depth_filters(self, depth_frame):
        """Apply RealSense filtering pipeline"""
        frame = depth_frame
        frame = self.th_filter.process(frame)
        frame = self.dec_filter.process(frame)
        frame = self.to_disparity.process(frame)
        frame = self.spa_filter.process(frame)
        frame = self.tmp_filter.process(frame)
        frame = self.to_depth.process(frame)
        frame = self.hole_filter.process(frame)
        depth_filtered = frame.as_depth_frame()
        return depth_filtered

    def apply_global_colormap(self, depth_m, min_d=0.5, max_d=4.0):
        depth_norm = np.clip((depth_m - min_d) / (max_d - min_d), 0.0, 1.0)
        depth_8u = (depth_norm * 255).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        return colored_rgb


def main():
    app = QApplication(sys.argv)

    setTheme(Theme.DARK)

    win = MainApp()
    win.show()
    win.setWindowTitle("Depth Estimation Performance on Mobile Robot")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()