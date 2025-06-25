import cv2
import threading
import logging
import socketserver
from http import server
import pyrealsense2 as rs
import numpy as np
import time

recording_enabled = False  # Global flag to control bag recording

class StreamingOutput:
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def update(self, frame):
        with self.condition:
            self.frame = frame
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = b"<html><body><img src='stream.mjpg' width='640' height='480'></body></html>"
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with self.server.output.condition:
                        self.server.output.condition.wait()
                        frame = self.server.output.frame
                    _, jpeg = cv2.imencode('.jpg', frame)
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    def __init__(self, server_address, handler_class, output):
        super().__init__(server_address, handler_class)
        self.output = output

def video_stream_fisheye(output):
    cap = cv2.VideoCapture(6)  # Sesuaikan index device
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output.update(gray)
        time.sleep(0.01)
    cap.release()

def video_stream_realsense(output_left, output_right):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    device = profile.get_device()
    recorder_started = False

    try:
        while True:
            global recording_enabled
            if recording_enabled and not recorder_started:
                try:
                    device.as_recordable_device().start_recording("records/cam2.bag")
                    print("Started recording cam2.bag")
                    recorder_started = True
                except Exception as e:
                    print("Failed to start recording:", e)

            if not recording_enabled and recorder_started:
                try:
                    device.as_recordable_device().stop_recording()
                    print("Stopped recording cam2.bag")
                    recorder_started = False
                except Exception as e:
                    print("Failed to stop recording:", e)

            frames = pipeline.wait_for_frames()
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            if left_frame and right_frame:
                left_image = np.asanyarray(left_frame.get_data())
                right_image = np.asanyarray(right_frame.get_data())
                output_left.update(left_image)
                output_right.update(right_image)
            time.sleep(0.01)
    finally:
        if recorder_started:
            device.as_recordable_device().stop_recording()
        pipeline.stop()

def start_streaming():
    output_webcam = StreamingOutput()
    output_left = StreamingOutput()
    output_right = StreamingOutput()

    threading.Thread(target=video_stream_fisheye, args=(output_webcam,), daemon=True).start()
    threading.Thread(target=video_stream_realsense, args=(output_left, output_right), daemon=True).start()

    server_webcam = StreamingServer(('', 8000), StreamingHandler, output_webcam)
    server_left = StreamingServer(('', 8001), StreamingHandler, output_left)
    server_right = StreamingServer(('', 8002), StreamingHandler, output_right)

    threading.Thread(target=server_webcam.serve_forever, daemon=True).start()
    threading.Thread(target=server_left.serve_forever, daemon=True).start()
    threading.Thread(target=server_right.serve_forever, daemon=True).start()

    print("Streaming started on ports: 8000 (webcam), 8001 (left), 8002 (right)")

def set_recording(enabled: bool):
    global recording_enabled
    recording_enabled = enabled