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

        encoder = self.ui.encoderBox.currentText()
        self.depth_model = DepthAnythingV2(encoder)
        self.depth_model.eval()
        self.depth_model.to('cpu') #TODO: Change to CUDA

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

    # def preprocess_for_depth_anything(self, frame):
    #     h, w, _ = frame.shape
    #
    #     encoder = self.ui.encoderBox.currentText()
    #
    #     if encoder == "vits": size = 364
    #     else: size = 518
    #
    #     resized = cv2.resize(frame, (size, size))
    #     return resized, (h, w)

    # def estimate_depth_anything(self, frame):
    #     resized, original_size = self.preprocess_for_depth_anything(frame)
    #     rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    #
    #     tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
    #     tensor = tensor / 255.0
    #     tensor = tensor.to('cpu')  #TODO: CHANGE TO CUDA
    #
    #     with torch.no_grad():
    #         depth = self.depth_model(tensor)[0]  # H x W depth map
    #
    #     depth = depth.cpu().numpy()
    #
    #     depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    #     depth_uint8 = (depth_norm * 255).astype(np.uint8)
    #
    #     depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    #     depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    #
    #     return depth_color

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

        except Exception as e:
            print("Error membuka .bag", e)
            self.rs_pipeline = None

        self.cap_cam = cv2.VideoCapture(cam_path)
        self.cap_ir1 = cv2.VideoCapture(ir1_path)
        self.cap_ir2 = cv2.VideoCapture(ir2_path)

        self.timer.start(33)

    def update_frames(self):
        self.update_frame(self.cap_cam, self.ui.camFrame)
        self.update_frame(self.cap_ir1, self.ui.intelLeftFrame)
        self.update_frame(self.cap_ir2, self.ui.intelRightFrame)
        self.update_depth_bag()

    def update_frame(self, cap, label):
        if cap is None or not cap.isOpened():
            label.setText("Tidak ada video")
            return

        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                label.setText("Tidak bisa membaca frame")
                return

        # self.update_frame_with_depth_anything(self.cap_cam, self.ui.depthFrameCam)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(label.width(), label.height())
        label.setPixmap(pix)

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

            # time.sleep(0.033)

            colorized = self.colorizer.colorize(depth_frame)
            depth_img = np.asanyarray(colorized.get_data())

            h, w, ch = depth_img.shape
            qimg = QImage(depth_img.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            pix = pix.scaled(self.ui.depthFrameIntel.width(),
                             self.ui.depthFrameIntel.height())
            self.ui.depthFrameIntel.setPixmap(pix)

        except Exception as e:
            print("Error membaca .bag:", e)

    # def update_frame_with_depth_anything(self, cap, label):
    #     if cap is None or not cap.isOpened():
    #         label.setText("Tidak ada video")
    #         return
    #
    #     ret, frame = cap.read()
    #
    #     if not ret:
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #         ret, frame = cap.read()
    #         if not ret:
    #             label.setText("Tidak bisa membaca frame")
    #             return
    #
    #     # Depth Performance estimation
    #     depth_frame = self.estimate_depth_anything(frame)
    #
    #     # Convert to QImage
    #     h, w, ch = depth_frame.shape
    #     qimg = QImage(depth_frame.data, w, h, w * ch, QImage.Format_RGB888)
    #     pix = QPixmap.fromImage(qimg)
    #     pix = pix.scaled(label.width(), label.height())
    #
    #     label.setPixmap(pix)

def main():
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
