# Depth program for prototyping

import os
import cv2
import numpy as np
import torch
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

from depth_anything_v2.dpt import DepthAnythingV2
from depth import Ui_Form  # asumsi UI mu sudah dibuat di depth.py


class DepthAnythingApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # Setup initial variables
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        # Model configs dictionary sama seperti di script CLI mu
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
        }

        # Load default model
        self.current_encoder = 'vits'
        self.depth_model = None
        self.load_model(self.current_encoder)

        # Video capture & state
        self.cap = None
        self.is_playing = False

        # Setup QTimer untuk baca frame tiap ~30ms (~33ms = 30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        # Connect signals
        self.ui.encoderBox.currentTextChanged.connect(self.on_encoder_changed)
        self.ui.loadDataset.clicked.connect(self.load_dataset)
        self.ui.playAndPause.clicked.connect(self.play_pause)

        self.ui.depthFrame0.setText("Entaniya")
        self.ui.depthFrame1.setText("RealSense")

    def load_model(self, encoder_name):
        print(f'Loading model {encoder_name}...')
        config = self.model_configs[encoder_name]
        self.depth_model = DepthAnythingV2(**config)
        checkpoint_path = f'checkpoints/depth_anything_v2_{encoder_name}.pth'
        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint not found: {checkpoint_path}')
            return
        self.depth_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.depth_model = self.depth_model.to(self.device).eval()

    def on_encoder_changed(self, encoder_name):
        if encoder_name in self.model_configs:
            self.current_encoder = encoder_name
            self.load_model(encoder_name)

    def load_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Videos")
        if folder:
            video_path = os.path.join(folder, 'cam1.avi')
            if os.path.isfile(video_path):
                if self.cap is not None:
                    self.cap.release()
                self.load_video(video_path)

            else:
                self.show_frame_placeholder()
                print("cam1.avi not found in selected folder.")

    def load_video(self, video_path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to open video {video_path}')
            self.cap = None
        else:
            self.ui.depthFrame0.setText("")  # Kosongkan tulisan No Video Loaded
            self.video_path = video_path

    def play_pause(self):
        if self.cap is None:
            return
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.ui.playAndPause.setText("Play")
        else:
            self.timer.start(33)  # ~30 FPS
            self.is_playing = True
            self.ui.playAndPause.setText("Pause")

    def next_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.is_playing = False
            self.ui.playAndPause.setText("Play")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart video
            return

        # Show raw frame di depthFrame0
        self.display_frame(self.ui.depthFrame0, frame)

        # Infer depth
        depth = self.depth_model.infer_image(frame, input_size=518)  # atau input_size diambil dari UI kalau perlu
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        # Color map depth
        cmap = cm.get_cmap('Spectral_r')
        depth_color = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)[:, :, ::-1]

        # Tampilkan depth map di depthFrame0 juga (bisa sesuaikan mau di sisi lain)
        self.display_frame(self.ui.depthFrame0, depth_color)

    def display_frame(self, widget, frame):
        """Convert cv2 image (BGR) to QPixmap and set to widget"""
        # Konversi BGR OpenCV ke RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(widget.width(), widget.height(), Qt.KeepAspectRatio)
        widget.setPixmap(pixmap)

    def show_frame_placeholder(self):
        # Kalau belum ada video, tampilkan layar kosong atau teks
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_img, "No Video Loaded", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.display_frame(self.ui.depthFrame0, blank_img)


if __name__ == '__main__':
    import sys
    from PyQt5 import QtWidgets
    import matplotlib.cm as cm

    app = QtWidgets.QApplication(sys.argv)
    window = DepthAnythingApp()
    window.show()
    sys.exit(app.exec_())
