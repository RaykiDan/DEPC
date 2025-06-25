from PyQt5.QtCore import QThread, pyqtSignal
import pyrealsense2 as rs
import numpy as np
import cv2
import time

class StereoBagProcessor(QThread):
    new_depth = pyqtSignal(object)
    print("Running stereo bag processor...")

    def __init__(self, cam2_bag_path, cam3_bag_path):
        super().__init__()
        self.cam2_path = cam2_bag_path
        self.cam3_path = cam3_bag_path
        self.running = True

    def run(self):
        # NOTE: cam3.bag is ignored for now, because RealSense stereo only needs one .bag file (with both IR)
        # Assuming cam2.bag includes both IR1 and IR2 (as recorded)

        cfg = rs.config()
        cfg.enable_device_from_file(self.cam2_path, repeat_playback=False)
        cfg.enable_stream(rs.stream.infrared, 1)
        cfg.enable_stream(rs.stream.infrared, 2)

        pipe = rs.pipeline()
        profile = pipe.start(cfg)

        depth_sensor = profile.get_device().first_depth_sensor()
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        align = rs.align(rs.stream.infrared)

        try:
            while self.running:
                frames = pipe.wait_for_frames()
                ir1 = frames.get_infrared_frame(1)
                ir2 = frames.get_infrared_frame(2)
                print("Got IR1 and IR2 frames")

                if not ir1 or not ir2:
                    continue

                img1 = np.asanyarray(ir1.get_data())
                img2 = np.asanyarray(ir2.get_data())

                # Pseudo-depth: abs diff + normalize (placeholder for real stereo matching)
                stereo = cv2.absdiff(img1, img2)
                stereo_norm = cv2.normalize(stereo, None, 0, 255, cv2.NORM_MINMAX)
                stereo_color = cv2.applyColorMap(stereo_norm.astype(np.uint8), cv2.COLORMAP_JET)

                self.new_depth.emit(stereo_color)
                time.sleep(0.05)

        finally:
            pipe.stop()

    def stop(self):
        self.running = False
        self.wait()