from PyQt5.QtCore import QThread, pyqtSignal
import cv2
from model.depth_estimator import DepthEstimator

class DepthProcessor(QThread):
    new_depth = pyqtSignal(object)

    def __init__(self, video_path, encoder='vits'):
        super().__init__()
        self.video_path = video_path
        self.encoder = encoder
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        model = DepthEstimator(encoder=self.encoder)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            depth_img = model.estimate_depth(frame)
            self.new_depth.emit(depth_img)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()