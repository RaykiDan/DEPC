# Program only for FPS testing

import os
import sys
import cv2
import numpy as np
import datetime
import threading
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QInputDialog
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from view import Ui_Form

class CameraStream(QThread):
    new_frame = pyqtSignal(object)  # buat ngirim frame ke MainWindow

    def __init__(self, url=None):
        super().__init__()
        self.url = url
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            print(f"Failed to open {self.url}")
            return

        self.running = True
        while self.running:
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.new_frame.emit(gray)  # kirim frame ke main program
            else:
                print(f"Failed to read frame from {self.url}")
                break
            self.msleep(10)  # biar nggak ngegas CPU 100%

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

def undistort_fisheye(frame): # TODO: Repair the fisheye undistort function
    DIM = (640, 480)  # Sesuaikan dengan resolusi frame Entaniya
    K = np.array([[400.0, 0.0, 320],
                  [0.0, 400.0, 480],
                  [0.0, 0.0, 1.0]])
    D = np.array([-0.28, 0.08, -0.001, 0.0003])  # Contoh, ganti dengan hasil kalibrasi

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted

def estimate_depth_from_entaniya(video_path):
    # Load frame pertama dan buat dummy depth
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return None

def estimate_depth_from_realsense_stereo(left_path, right_path):
    capL = cv2.VideoCapture(left_path)
    capR = cv2.VideoCapture(right_path)
    retL, left = capL.read()
    retR, right = capR.read()
    capL.release()
    capR.release()
    if retL and retR:
        # Sederhanakan dengan perbedaan kanal sebagai dummy
        stereo_diff = cv2.absdiff(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
        return cv2.applyColorMap(stereo_diff, cv2.COLORMAP_JET)
    return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.monitor_active = True

        self.cams = [None, None, None]  # entaniya, intel left, intel right
        self.frames = [None, None, None]  # simpan frame baru tiap kamera
        self.ip_inputs = ["", "", ""]

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)

        self.setup_timers()

        # IP masing-masing kamera
        self.ui.loadFisheye.clicked.connect(lambda: self.set_ip(0))
        self.ui.loadLeft.clicked.connect(lambda: self.set_ip(1))
        self.ui.loadRight.clicked.connect(lambda: self.set_ip(2))

        # tombol
        self.ui.recordMonitorButton.clicked.connect(self.toggle_recording)
        # self.ui.clearMonitorButton.clicked.connect(self.clear_monitor_views) # TODO: Buat function untuk menghapus monitor yang sedang ditampilkan
        # self.ui.loadDataset.clicked.connect(self.load_dataset) # TODO: Persiapkan tombol load dataset

        self.is_recording = False
        self.recorders = [None, None, None]
        self.frames = [None, None, None]
        self.record_duration = 0
        self.frames_received = [0, 0, 0]  # Untuk FPS
        self.fps_labels = [self.ui.fpsLabel1, self.ui.fpsLabel2, self.ui.fpsLabel3]

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
        if idx == 0:
            # Kamera Entaniya → undistort
            frame = undistort_fisheye(frame)

        self.frames[idx] = frame
        self.frames_received[idx] += 1

        if self.is_recording and self.recorders[idx] is not None:
            self.recorders[idx].write(frame)

    def update_frames(self):
        labels = [self.ui.entaniyaFrame, self.ui.intelLeftFrame, self.ui.intelRightFrame]

        label_widths = [label.width() for label in labels]
        label_heights = [label.height() for label in labels]
        target_width = min(label_widths)
        target_height = min(label_heights)

        for frame, label in zip(self.frames, labels):
            if frame is not None:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_img)
                scaled_pixmap = pixmap.scaled(target_width, target_height, Qt.KeepAspectRatio)
                label.setPixmap(scaled_pixmap)
            else:
                label.clear()

    def start_recording(self):
        if not any(frame is not None for frame in self.frames):
            print("No frames available yet.")
            return

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        # Menyimpan rekaman di folder "records"
        output_filenames = [
            os.path.join("records", "cam1.avi"),
            os.path.join("records", "cam2.avi"),
            os.path.join("records", "cam3.avi")
        ]
        os.makedirs("records", exist_ok=True)  # Buat folder records kalau belum ada

        # TODO: Buat agar rekaman dibuatkan folder tersendiri berdasarkan waktu dimulainya rekaman, sama seperti untuk image sequence

        width = 640
        height = 480
        fps = 30

        self.recorders = []
        for idx, frame in enumerate(self.frames):
            if frame is not None:
                writer = cv2.VideoWriter(output_filenames[idx], fourcc, fps, (width, height))
                self.recorders.append(writer)
            else:
                self.recorders.append(None)

        self.is_recording = True
        self.record_duration = 0

        # Ubah warna tombol Record
        self.ui.recordMonitorButton.setStyleSheet("background-color: red")

        print("Recording started.")

    def stop_recording(self):
        self.is_recording = False
        for recorder in self.recorders:
            if recorder:
                recorder.release()

        self.ui.recordMonitorButton.setStyleSheet("")
        print("Recording stopped.")

        # Mulai proses video to images
        self.process_recorded_videos()

    def process_recorded_videos(self):
        output_filenames = [
            os.path.join("records", "cam1.avi"),
            os.path.join("records", "cam2.avi"),
            os.path.join("records", "cam3.avi")
        ]
        cam_names = ["cam1", "cam2", "cam3"]

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # bikin folder berdasarkan waktu saat ini
        os.makedirs("image_sequence", exist_ok=True)  # bikin folder image_sequence kalau belum ada
        base_output_dir = os.path.join("image_sequence", timestamp)

        for idx, video_file in enumerate(output_filenames):
            if not os.path.exists(video_file):
                print(f"Video file {video_file} not found, skipping.")
                continue

            cam_folder = os.path.join(base_output_dir, cam_names[idx])
            video_to_images(video_file, cam_folder)

    def setup_timers(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)

        self.record_timer = QTimer()
        self.record_timer.timeout.connect(self.update_record_status)
        self.record_timer.start(1000)

    def update_record_status(self):
        if self.is_recording:
            self.record_duration += 1
            # self.ui.recordTimeLabel.setText(f"{self.record_duration} s") # TODO: Buat tanda sedang merekam

        # Update FPS display
        for idx in range(3):
            fps = self.frames_received[idx]
            self.frames_received[idx] = 0  # Reset
            self.fps_labels[idx].setText(f"FPS: {fps}")

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

def video_to_images(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames to {output_dir}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
