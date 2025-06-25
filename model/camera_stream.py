import cv2
from PyQt5.QtCore import QThread, pyqtSignal

class CameraStream(QThread):
    new_frame = pyqtSignal(object)  # Sinyal untuk mengirim frame ke UI

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
                self.new_frame.emit(frame)
            else:
                print(f"Failed to read frame from {self.url}")
                break
            self.msleep(10)  # Biar nggak full CPU

        cap.release()

    def stop(self):
        self.running = False
        self.wait()