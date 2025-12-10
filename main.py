import sys

import cv2
import time
import torch
import pyrealsense2 as rs
import numpy as np

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ui.interface2 import Ui_Form
from depth_anything_v2.dpt import DepthAnythingV2

def generate_depth_ruler(width=40, height=250, dmin=0.5, dmax=3.5):
    ruler = np.zeros((height, width, 3), dtype=np.uint8)

    # vertical gradient: index 0 = top (red), index height-1 = bottom (blue)
    gradient = np.linspace(1.0, 0.0, height).astype(np.float32)
    gradient_8u = (gradient * 255).astype(np.uint8)
    turbo = cv2.applyColorMap(gradient_8u, cv2.COLORMAP_TURBO)

    # fill ruler
    for y in range(height):
        ruler[y, :] = turbo[y]

    # Ticks: gunakan arange untuk menghindari akumulasi float
    step = 0.5
    ticks = np.arange(dmin, dmax + 1e-6, step)
    for value in ticks:
        # pastikan y berada di rentang [0, height-1]
        rel = (value - dmin) / (dmax - dmin)
        y = int(round((height - 1) * (1.0 - rel)))
        y = np.clip(y, 0, height - 1)

        cv2.line(ruler, (0, y), (12, y), (0, 0, 0), 1)

        # label hanya untuk integer meter (mis. 1.0, 2.0, ...)
        if abs(value - round(value)) < 1e-6:
            cv2.putText(
                ruler,
                f"{int(round(value))}m",
                (15, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

    return ruler

def generate_horizontal_ruler(width=120, height=20):
    # buat gradient horizontal 0..255
    grad = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    grad = np.flip(grad, axis=1)

    # apply colormap JET -> returns BGR
    ruler_bgr = cv2.applyColorMap(grad, cv2.COLORMAP_JET)

    # convert to RGB because kamu memakai QImage.Format_RGB888 di tempat lain
    ruler_rgb = cv2.cvtColor(ruler_bgr, cv2.COLOR_BGR2RGB)
    return ruler_rgb

class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.playing = False
        self.loaded = False
        self.caps = {}  # cam1, cam2, cam3

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
        self.th_filter.set_option(rs.option.min_distance, 0.2)  # meter
        self.th_filter.set_option(rs.option.max_distance, 3.0)  # meter

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
            "vitl": 252,  # Sesuaikan kebutuhan
            "default": 252
        }

        self.checkpoint_paths = {
            "vits": "checkpoints/depth_anything_v2_vits.pth",
            "vitb": "checkpoints/depth_anything_v2_vitb.pth",
            "vitl": "checkpoints/depth_anything_v2_vitl.pth",
        }

        ruler = generate_horizontal_ruler(
            width=self.ui.depthObjectRuler.width(),
            height=self.ui.depthObjectRuler.height()
        )

        h, w, c = ruler.shape
        qimg = QImage(ruler.data, w, h, 3 * w, QImage.Format_BGR888)
        self.ui.depthObjectRuler.setPixmap(QPixmap.fromImage(qimg))

    def toggle_play(self):
        if not self.loaded:
            print("[WARN] Belum ada dataset. Tekan Select dulu.")
            return

        self.playing = not self.playing

        if self.playing:
            print("[INFO] PLAY")
            self.timer.start(33)  # resume
            self.ui.startAndStopButton.setText("Stop")
        else:
            print("[INFO] PAUSE")
            self.timer.stop()  # pause
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

        # Stop realsense pipeline
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

        # self.ui.depthValueCam.setText("-")
        # self.ui.depthValueIntel.setText("-")

        self.ui.startAndStopButton.setText("Start")

    def update_depth_ruler(self):
        ruler_img = generate_depth_ruler(40, 250, dmin=0.5, dmax=3.5)
        # ruler_img dari generate_depth_ruler sudah BGR — ubah ke RGB agar konsisten
        ruler_img_rgb = cv2.cvtColor(ruler_img, cv2.COLOR_BGR2RGB)

        h, w, ch = ruler_img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(ruler_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        self.ui.depthRulerCam.setPixmap(pixmap)
        self.ui.depthRulerIntel.setPixmap(pixmap)
        self.ui.depthRulerIntel_2.setPixmap(pixmap)

        # horizontal label / object ruler
        label_img = generate_horizontal_ruler(width=120, height=20)
        hl_h, hl_w, hl_ch = label_img.shape
        hl_bpl = hl_ch * hl_w
        qimg2 = QImage(label_img.data, hl_w, hl_h, hl_bpl, QImage.Format_RGB888)
        pix2 = QPixmap.fromImage(qimg2)
        # Ganti nama widget ini sesuai UI-mu (contoh: depthObjectRuler)
        if hasattr(self.ui, "depthObjectRuler"):
            self.ui.depthObjectRuler.setPixmap(pix2)

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
        return float(raw_depth_map[cy, cx])  # dalam meter

    def estimate_depth_anything(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_depth = self.infer_dav2(rgb)

        if raw_depth is None:
            return np.zeros_like(frame), None

        # raw_depth (meter) → colormap global
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

                center_depth = self.get_center_depth_dav2(raw_depth)

                # self.ui.depthValueCam.setText(f"{center_depth: .3f} m")

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

            # Ambil nilai tengah (filtered)
            center_depth_m = self.get_center_depth_realsense(filtered)
            # self.ui.depthValueIntel.setText(f"{center_depth_m: .3f} m")

            depth_raw = np.asanyarray(filtered.get_data()).astype(np.float32) * 0.001  # mm → meter
            depth_img = self.apply_global_colormap(depth_raw)

            # Tampilkan
            h, w = depth_img.shape[:2]
            bytes_per_line = depth_img.strides[0]
            qimg = QImage(depth_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(self.ui.depthFrameIntel.width(),
                                                 self.ui.depthFrameIntel.height())
            self.ui.depthFrameIntel.setPixmap(pix)

        except Exception as e:
            print("[ERROR] Gagal membaca .bag:", e)

    def apply_depth_filters(self, depth_frame):
        """
        Apply RealSense filtering pipeline and ensure output is depth_frame.
        """

        frame = depth_frame

        # 1. Threshold
        frame = self.th_filter.process(frame)

        # 2. Decimation
        frame = self.dec_filter.process(frame)

        # 3. Depth → disparity
        frame = self.to_disparity.process(frame)

        # 4. Spatial filter
        frame = self.spa_filter.process(frame)

        # 5. Temporal filter
        frame = self.tmp_filter.process(frame)

        # 6. Disparity → depth
        frame = self.to_depth.process(frame)

        # 7. Hole filling
        frame = self.hole_filter.process(frame)

        # <-- FIX PENTING!
        # Pastikan kembali jadi depth frame
        depth_filtered = frame.as_depth_frame()
        return depth_filtered

    def apply_global_colormap(self, depth_m, min_d=0.5, max_d=3.5):
        # normalisasi ke [0,1] where 0 = min_d (near), 1 = max_d (far)
        depth_norm = np.clip((depth_m - min_d) / (max_d - min_d), 0.0, 1.0)

        # JANGAN membalikkan (1 - depth_norm) — kita ingin 0->blue, 1->red for Turbo
        depth_8u = (depth_norm * 255).astype(np.uint8)

        # apply colormap (returns BGR)
        colored_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)

        # convert to RGB for QImage.Format_RGB888
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        return colored_rgb

def main():
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    win.setWindowTitle("Depth Estimation Performance on Mobile Robot")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
