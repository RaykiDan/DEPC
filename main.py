import sys

import cv2
import time
import torch
import pyrealsense2 as rs
import numpy as np

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from interface import Ui_Form
from depth_anything_v2.dpt import DepthAnythingV2

class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

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

        self.ui.selectButton.clicked.connect(self.select_folder)

        self.cap_cam = None
        self.cap_ir1 = None
        self.cap_ir2 = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)

        self.encoder_to_size = {
            "vitl": 252,  # dari 518 → 320 (FPS naik drastis)
            "default": 252
        }

        self.checkpoint_paths = {
            "vits": "checkpoints/depth_anything_v2_vits.pth",
            "vitb": "checkpoints/depth_anything_v2_vitb.pth",
            "vitl": "checkpoints/depth_anything_v2_vitl.pth",
        }

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
                # Load model with correct config
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

    def preprocess_for_depth_anything(self, frame):
        h, w = frame.shape[:2]

        encoder = self.ui.encoderBox.currentText() if hasattr(self.ui, "encoderBox") else "vits"
        size = self.encoder_to_size.get(encoder, self.encoder_to_size["default"])

        resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        rgb = (rgb-0.5)/0.5

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        return tensor, (h, w)

    def infer_dav2(self, frame):
        """
        Gunakan pipeline metric bawaan DepthAnythingV2:
        image2tensor() → forward() → interpolate ke ukuran asli.
        """
        self.ensure_model_loaded()
        try:
            raw_depth = self.depth_model.infer_image(frame)
        except Exception as e:
            print("[ERROR] infer_image metric DAV2:", e)
            return None

        return raw_depth  # sudah dalam satuan meter

    def get_center_depth_realsense(self, depth_frame):
        w = depth_frame.get_width()
        h = depth_frame.get_height()
        cx, cy = w // 2, h // 2
        depth_value = depth_frame.get_distance(cx, cy)  # dalam meter
        return depth_value

    def get_center_depth_dav2(self, raw_depth_map):
        h, w = raw_depth_map.shape
        cx, cy = w // 2, h // 2
        return float(raw_depth_map[cy, cx])  # DALAM METER

    def estimate_depth_anything(self, frame):
        raw_depth = self.infer_dav2(frame)
        if raw_depth is None:
            return np.zeros_like(frame), None

        # raw_depth DI SINI sudah real depth dalam meter
        dmin, dmax = raw_depth.min(), raw_depth.max()

        depth_norm = (raw_depth - dmin) / (dmax - dmin + 1e-6)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)

        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
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
            # self.rs_config.enable_stream(rs.stream.depth, rs.format.z16, 30)

            self.rs_profile = self.rs_pipeline.start(self.rs_config)

            playback = self.rs_profile.get_device().as_playback()
            playback.set_real_time(False)
            print("[INFO] Berhasil membuka .bag")

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

    def update_frames(self):
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
                depth_color, raw_depth = self.estimate_depth_anything(frame)

                h2, w2 = depth_color.shape[:2]
                bytes_per_line2 = depth_color.strides[0]
                qimg2 = QImage(depth_color.data, w2, h2, bytes_per_line2, QImage.Format_RGB888)
                pix2 = QPixmap.fromImage(qimg2).scaled(depth_label.width(), depth_label.height())
                depth_label.setPixmap(pix2)

                center_depth = self.get_center_depth_dav2(raw_depth)

                self.ui.depthValueCam.setText(f"{center_depth: .3f} m")

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

            center_depth_m = self.get_center_depth_realsense(depth_frame)
            self.ui.depthValueIntel.setText(f"{center_depth_m: .3f} m")

            # time.sleep(0.033)

            colorized = self.colorizer.colorize(depth_frame)
            depth_img = np.asanyarray(colorized.get_data())

            h, w = depth_img.shape [:2]
            bytes_per_line = depth_img.strides[0]
            qimg = QImage(depth_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(self.ui.depthFrameIntel.width(),
                                                 self.ui.depthFrameIntel.height())
            self.ui.depthFrameIntel.setPixmap(pix)

        except Exception as e:
            print("[ERROR] Gagal membaca .bag:", e)

def main():
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
