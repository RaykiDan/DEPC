import pyrealsense2 as rs
import time

class BagRecorder:
    def __init__(self, output_path='records/cam2.bag'):
        self.output_path = output_path
        self.pipeline = None
        self.running = False

    def start(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_record_to_file(self.output_path)

        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

        print(f"[BagRecorder] Starting: {self.output_path}")
        self.pipeline.start(config)
        self.running = True

        try:
            while self.running:
                self.pipeline.wait_for_frames()
                time.sleep(0.01)  # biar tidak full CPU
        except Exception as e:
            print(f"[BagRecorder] Error: {e}")
        finally:
            print(f"[BagRecorder] Stopping: {self.output_path}")
            self.pipeline.stop()

    def stop(self):
        self.running = False
