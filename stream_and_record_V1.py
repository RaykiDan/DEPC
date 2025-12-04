import os, cv2, time, queue, signal, threading
import numpy as np
import pyrealsense2 as rs
from collections import deque

# ---------- CONFIG ----------
WIDTH, HEIGHT, FPS = 640, 480, 30
WEBCAM_INDEX = 8
COLOR_CODEC = "XVID" #format video
OUTPUT_DIR = "records"
DEPTH_ALPHA = 0.05
ENABLE_ALIGN = True
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
frame_q = queue.Queue(maxsize=4)
write_q = queue.Queue(maxsize=64)
stop_event = threading.Event()
fps_ts = deque(maxlen=FPS * 10)

# -------- Recording .bag --------
def start_realsense_with_recording(bag_path):
    global pipeline, config, align
    pipeline.stop()

    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    cfg.enable_record_to_file(bag_path)

    align = rs.align(rs.stream.color) if ENABLE_ALIGN else None
    pipeline.start(cfg)

# ---------- RealSense ----------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
align = rs.align(rs.stream.color) if ENABLE_ALIGN else None

def grabber():
    try: pipeline.start(config)
    except Exception as e: print("[grabber] start failed:", e); stop_event.set(); return
    print("[grabber] started")

    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=500)
            if align: frames = align.process(frames)
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if depth and color:
                frame_q.put((time.time(),
                np.asanyarray(color.get_data()),
                np.asanyarray(depth.get_data())),
                block=False)
        except: pass

    print("[grabber] stopping")
    pipeline.stop()
    stop_event.set()

# ---------- Writer ----------
def writer():
    wr_rgb = wr_depth = wr_web = None
    recording = False
    print("[writer] started")

    while True:
        item = write_q.get()
        if item is None: break
        cmd = item["cmd"]

        if cmd == "start" and not recording:
            p_rgb, p_d, p_w = item["rgb"], item["depth"], item["web"]
            fourcc = cv2.VideoWriter_fourcc(*COLOR_CODEC)
            wr_rgb  = cv2.VideoWriter(p_rgb,  fourcc, FPS, (WIDTH, HEIGHT))
            wr_depth= cv2.VideoWriter(p_d,    fourcc, FPS, (WIDTH, HEIGHT))
            wr_web  = cv2.VideoWriter(p_w,    fourcc, FPS, (WIDTH, HEIGHT))
            recording = True
            print("[writer] recording →", p_rgb)

        elif cmd == "frame" and recording:
            wr_rgb.write(item["rgb"])
            wr_depth.write(item["depth"])
            wr_web.write(item["web"])

        elif cmd == "stop" and recording:
            wr_rgb.release(); wr_depth.release(); wr_web.release()
            recording = False
            print("[writer] stopped")

        elif cmd == "exit":
            if wr_rgb: wr_rgb.release()
            if wr_depth: wr_depth.release()
            if wr_web: wr_web.release()
            print("[writer] exit"); break

    print("[writer] done")

# ---------- Utils ----------
def depth_to_color(d16):
    d8 = cv2.convertScaleAbs(d16, alpha=DEPTH_ALPHA)
    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)

# ---------- Main ----------
def main():
    cam = cv2.VideoCapture(WEBCAM_INDEX)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cam.set(cv2.CAP_PROP_FPS, FPS)

    threading.Thread(target=grabber, daemon=True).start()
    threading.Thread(target=writer, daemon=True).start()

    recording, last_fps = False, time.time()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    while not stop_event.is_set():
        try: ts, rgb, depth = frame_q.get(timeout=0.5)
        except queue.Empty: continue
        fps_ts.append(time.time())

        ret, web = cam.read()
        if not ret: web = np.zeros_like(rgb)

        dcolor = depth_to_color(depth)

        if recording:
            cv2.circle(rgb, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(rgb, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            try: write_q.put_nowait({"cmd":"frame","rgb":rgb.copy(),"depth":dcolor.copy(),"web":web.copy()})
            except: pass

        cv2.imshow("RealSense RGB", rgb)
        cv2.imshow("RealSense Depth Performance", dcolor)
        cv2.imshow("Webcam", web)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27): stop_event.set()
        elif k == ord('r'):
            if not recording:
                ts = time.strftime("%d%m%Y-%H%M%S")
                bag_path = os.path.join(OUTPUT_DIR, f"session_{ts}.bag")

                start_realsense_with_recording(bag_path)

                write_q.put({
                    "cmd": "start",
                    "rgb": f"{OUTPUT_DIR}/rgb_{ts}.avi",
                    "depth": f"{OUTPUT_DIR}/depth_{ts}.avi",
                    "web": f"{OUTPUT_DIR}/web_{ts}.avi"
                })
                print("[main] REC start")
            else:
                write_q.put({"cmd": "stop"})
                pipeline.stop()
                time.sleep(0.4)
                pipeline.start(config)
                print("[main] REC stop")
            recording = not recording

        if time.time() - last_fps >= 5:
            if len(fps_ts) > 1:
                fps = (len(fps_ts)-1)/(fps_ts[-1]-fps_ts[0])
                print(f"[fps] {fps:.2f}, fq={frame_q.qsize()}, wq={write_q.qsize()}")
            last_fps = time.time()

    print("[main] exit")
    write_q.put({"cmd": "exit"})
    write_q.put(None)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
