import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

# Simpan ke file .bag
config.enable_record_to_file("output.bag")

# Konfigurasi stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    print("Merekam... Tekan Ctrl+C untuk berhenti")
    while True:
        frames = pipeline.wait_for_frames()
        # (Jika ingin, proses frame di sini)
except KeyboardInterrupt:
    print("Stop recording")
finally:
    pipeline.stop()
