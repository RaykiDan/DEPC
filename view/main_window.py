import os
import cv2
import datetime
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QInputDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from view.view import Ui_Form
from model.camera_stream import CameraStream
from model.depth_estimator import DepthEstimator
from model.stream_all import set_recording

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.cams = [None, None, None]
        self.frames = [None, None, None]
        self.recorders = [None, None, None]
        self.frames_received = [0, 0, 0]
        self.fps_labels = [self.ui.fpsLabel1, self.ui.fpsLabel2, self.ui.fpsLabel3]

        self.is_recording = False
        self.record_duration = 0

        self.depth_estimator = DepthEstimator()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)

        self.record_timer = QTimer()
        self.record_timer.timeout.connect(self.update_record_status)
        self.record_timer.start(1000)

        self.ui.loadFisheye.clicked.connect(lambda: self.set_ip(0))
        self.ui.loadLeft.clicked.connect(lambda: self.set_ip(1))
        self.ui.loadRight.clicked.connect(lambda: self.set_ip(2))

        self.ui.recordMonitorButton.clicked.connect(self.toggle_recording)
        self.ui.loadDataset.clicked.connect(self.handle_depth_dataset)

    def set_ip(self, cam_idx):
        ip, ok = QInputDialog.getText(self, f"Input IP Kamera {cam_idx + 1}", "Masukkan IP:",
                                      text="http://localhost:8000/stream.mjpg")
        if ok and ip:
            if self.cams[cam_idx]:
                self.cams[cam_idx].stop()

            cam = CameraStream(ip)
            cam.new_frame.connect(lambda frame, idx=cam_idx: self.update_frame(idx, frame))
            cam.start()

            self.cams[cam_idx] = cam
            print(f"Camera {cam_idx + 1} streaming from {ip}")

    def update_frame(self, idx, frame):
        self.frames[idx] = frame
        self.frames_received[idx] += 1
        if self.is_recording and self.recorders[idx]:
            self.recorders[idx].write(frame)

    def update_frames(self):
        labels = [self.ui.entaniyaFrame, self.ui.intelLeftFrame, self.ui.intelRightFrame]
        for frame, label in zip(self.frames, labels):
            if frame is not None:
                self.display_image_on_label(label, frame)
            else:
                label.clear()

    def display_image_on_label(self, label, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(scaled_pixmap)

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if not any(frame is not None for frame in self.frames):
            print("No frames available yet.")
            return

        os.makedirs("records", exist_ok=True)
        avi_path = "records/cam1.avi"
        width, height, fps = 640, 480, 30
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        # Rekam hanya Entaniya (cam1) ke AVI
        if self.frames[0] is not None:
            self.recorders[0] = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
        else:
            self.recorders[0] = None

        # Tandai RealSense (cam2, cam3) tidak rekam AVI
        self.recorders[1] = None
        self.recorders[2] = None

        set_recording(True)  # Mulai rekam .bag dari RealSense

        self.is_recording = True
        self.record_duration = 0
        self.ui.recordMonitorButton.setStyleSheet("background-color: red")
        print("Recording started.")

    def stop_recording(self):
        set_recording(False)  # Hentikan rekam .bag dari RealSense

        for recorder in self.recorders:
            if recorder:
                recorder.release()

        self.is_recording = False
        self.ui.recordMonitorButton.setStyleSheet("")
        print("Recording stopped.")
        self.process_recorded_videos()

    def process_recorded_videos(self):
        filenames = [f"records/cam{i+1}.avi" for i in range(3)]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = os.path.join("image_sequence", timestamp)
        os.makedirs(base_dir, exist_ok=True)

        for i, video in enumerate(filenames):
            if not os.path.exists(video):
                continue
            self.video_to_images(video, os.path.join(base_dir, f"cam{i+1}"))

    def video_to_images(self, video_path, output_dir):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            return

        os.makedirs(output_dir, exist_ok=True)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(output_dir, f"frame_{idx:04d}.jpg"), frame)
            idx += 1
        cap.release()
        print(f"Extracted {idx} frames to {output_dir}")

    def handle_depth_dataset(self):
        video_path = QFileDialog.getExistingDirectory(self, "Select Folder", "records")
        if not video_path:
            return

        cam1_path = os.path.join(video_path, "cam1.avi")
        cam2_path = os.path.join(video_path, "cam2.bag")
        cam3_path = os.path.join(video_path, "cam3.bag")

        if not all(os.path.exists(p) for p in [cam1_path, cam2_path, cam3_path]):
            QtWidgets.QMessageBox.critical(self, "Error", "cam1.avi, cam2.bag, atau cam3.bag tidak ditemukan.")
            return

        if hasattr(self, 'depth_worker') and self.depth_worker.isRunning():
            self.depth_worker.stop()

        from model.depth_worker import DepthProcessor
        encoder = self.ui.encoderBox.currentText()
        self.depth_worker = DepthProcessor(cam1_path, encoder)
        self.depth_worker.new_depth.connect(lambda img: self.display_image_on_label(self.ui.depthFrame0, img))
        self.depth_worker.start()

        from model.stereo_bag_worker import StereoBagProcessor
        if hasattr(self, 'stereo_worker') and self.stereo_worker.isRunning():
            self.stereo_worker.stop()

        self.stereo_worker = StereoBagProcessor(cam2_path, cam3_path)
        self.stereo_worker.new_depth.connect(lambda img: self.display_image_on_label(self.ui.depthFrame1, img))
        self.stereo_worker.start()

    def update_record_status(self):
        if self.is_recording:
            self.record_duration += 1
        for idx in range(3):
            fps = self.frames_received[idx]
            self.frames_received[idx] = 0
            self.fps_labels[idx].setText(f"FPS: {fps}")
