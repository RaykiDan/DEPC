import os, cv2, time, queue, signal, threading
import numpy as np
import pyrealsense2 as rs
from collections import deque

# ---------- CONFIG ----------
WIDTH, HEIGHT, FPS = 640, 480, 30
WEBCAM_INDEX = 8
COLOR_CODEC = "XVID"
OUTPUT_DIR = "records"
DEPTH_ALPHA = 0.08
ENABLE_ALIGN = False   # keep False for IR-only
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
frame_q = queue.Queue(maxsize=8)
write_q = queue.Queue(maxsize=256)
stop_event = threading.Event()
fps_ts = deque(maxlen=FPS * 10)

# ---------- RealSense helpers ----------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)
config.enable_stream(rs.stream.infrared, 2, WIDTH, HEIGHT, rs.format.y8, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)  # [NEW] RealSense RGB

# ---------- RealSense Filters ----------
dec_filter  = rs.decimation_filter()
spa_filter  = rs.spatial_filter()
tmp_filter  = rs.temporal_filter()
hole_filter = rs.hole_filling_filter()

# Disparity transforms
to_disparity   = rs.disparity_transform(True)
to_depth       = rs.disparity_transform(False)

# Threshold filter
th_filter = rs.threshold_filter()
th_filter.set_option(rs.option.min_distance, 0.1)
th_filter.set_option(rs.option.max_distance, 4.0)

def apply_depth_filters(depth):
    depth = th_filter.process(depth)
    depth = dec_filter.process(depth)
    depth = to_disparity.process(depth)
    depth = spa_filter.process(depth)
    depth = tmp_filter.process(depth)
    depth = to_depth.process(depth)
    depth = hole_filter.process(depth)
    return depth

def start_realsense_with_recording(bag_path):
    global pipeline, config
    try:
        pipeline.stop()
    except Exception:
        pass
    time.sleep(0.35)

    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)
    cfg.enable_stream(rs.stream.infrared, 2, WIDTH, HEIGHT, rs.format.y8, FPS)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)  # [NEW]
    cfg.enable_record_to_file(bag_path)

    pipeline.start(cfg)

# ---------- Grabber ----------
def grabber():
    try:
        pipeline.start(config)
    except Exception as e:
        print("[grabber] start failed:", e)
        stop_event.set()
        return
    print("[grabber] started")

    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=500)
            depth  = frames.get_depth_frame()
            ir1    = frames.get_infrared_frame(1)
            ir2    = frames.get_infrared_frame(2)
            color  = frames.get_color_frame()          # [NEW]

            if depth and ir1 and ir2 and color:        # [MODIFIED] include color check
                depth = apply_depth_filters(depth)

                depth_np = np.asanyarray(depth.get_data())
                ir1_np   = np.asanyarray(ir1.get_data())
                ir2_np   = np.asanyarray(ir2.get_data())
                color_np = np.asanyarray(color.get_data())  # [NEW] already BGR

                frame_q.put((time.time(), ir1_np, ir2_np, depth_np, color_np),  # [MODIFIED]
                            block=False)
        except queue.Full:
            pass
        except Exception:
            pass

    try:
        pipeline.stop()
    except Exception:
        pass
    stop_event.set()
    print("[grabber] exited")

# ---------- Writer ----------
def writer():
    wr_ir1 = wr_ir2 = wr_depth = wr_web = wr_rs_color = None  # [MODIFIED]
    recording = False
    print("[writer] started")

    while True:
        item = write_q.get()
        if item is None:
            break

        try:
            cmd = item.get("cmd")
            if cmd == "start" and not recording:
                p_ir1 = item["infrared1"]
                p_ir2 = item["infrared2"]
                p_d   = item["depth"]
                p_w   = item["web"]
                p_rc  = item["rs_color"]               # [NEW]
                fourcc = cv2.VideoWriter_fourcc(*COLOR_CODEC)
                wr_ir1     = cv2.VideoWriter(p_ir1, fourcc, FPS, (WIDTH, HEIGHT))
                wr_ir2     = cv2.VideoWriter(p_ir2, fourcc, FPS, (WIDTH, HEIGHT))
                wr_depth   = cv2.VideoWriter(p_d,   fourcc, FPS, (WIDTH, HEIGHT))
                wr_web     = cv2.VideoWriter(p_w,   fourcc, FPS, (WIDTH, HEIGHT))
                wr_rs_color = cv2.VideoWriter(p_rc, fourcc, FPS, (WIDTH, HEIGHT))  # [NEW]
                all_open = (wr_ir1.isOpened() and wr_ir2.isOpened() and
                            wr_depth.isOpened() and wr_web.isOpened() and
                            wr_rs_color.isOpened())    # [MODIFIED]
                if not all_open:
                    print("[writer] failed to open writers")
                    for w in [wr_ir1, wr_ir2, wr_depth, wr_web, wr_rs_color]:
                        if w: w.release()
                    recording = False
                else:
                    recording = True
                    print("[writer] recording ->", p_ir1, p_ir2, p_d, p_w, p_rc)

            elif cmd == "frame" and recording:
                wr_ir1.write(item["infrared1"])
                wr_ir2.write(item["infrared2"])
                wr_depth.write(item["depth"])
                wr_web.write(item["web"])
                wr_rs_color.write(item["rs_color"])    # [NEW]

            elif cmd == "stop" and recording:
                for w in [wr_ir1, wr_ir2, wr_depth, wr_web, wr_rs_color]:
                    if w: w.release()                  # [MODIFIED]
                recording = False
                print("[writer] stopped")

            elif cmd == "exit":
                for w in [wr_ir1, wr_ir2, wr_depth, wr_web, wr_rs_color]:
                    if w: w.release()                  # [MODIFIED]
                print("[writer] exiting")
                break
        except Exception as e:
            print("[writer] exception:", e)

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

    t_grab  = threading.Thread(target=grabber, daemon=True)
    t_write = threading.Thread(target=writer,  daemon=True)
    t_grab.start()
    t_write.start()

    recording = False
    last_fps  = time.time()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    while not stop_event.is_set():
        try:
            ts, ir1, ir2, depth, rs_color = frame_q.get(timeout=0.5)  # [MODIFIED]
        except queue.Empty:
            continue

        fps_ts.append(time.time())
        ret, web = cam.read()
        if not ret:
            web = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        dcolor   = depth_to_color(depth)
        vis_ir1  = cv2.cvtColor(ir1, cv2.COLOR_GRAY2BGR)
        vis_ir2  = cv2.cvtColor(ir2, cv2.COLOR_GRAY2BGR)

        if recording:
            cv2.circle(vis_ir1, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(vis_ir1, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # show windows
        cv2.imshow("RealSense IR1",              vis_ir1)
        cv2.imshow("RealSense IR2",              vis_ir2)
        cv2.imshow("RealSense Depth",            dcolor)
        cv2.imshow("RealSense Color",            rs_color)   # [NEW]
        cv2.imshow("Webcam (Logitech C930)",     web)        # [MODIFIED label]

        if recording:
            try:
                write_q.put_nowait({
                    "cmd":       "frame",
                    "infrared1": vis_ir1.copy(),
                    "infrared2": vis_ir2.copy(),
                    "depth":     dcolor.copy(),
                    "web":       web.copy(),
                    "rs_color":  rs_color.copy()             # [NEW]
                })
            except queue.Full:
                print("[main] write_q full, dropping frame")

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            stop_event.set()
            break

        elif k == ord('r'):
            if not recording:
                tsname   = time.strftime("%Y%m%d-%H%M%S")
                bag_path = os.path.join(OUTPUT_DIR, f"session_{tsname}.bag")
                start_realsense_with_recording(bag_path)
                write_q.put({
                    "cmd":       "start",
                    "infrared1": f"{OUTPUT_DIR}/ir1_{tsname}.avi",
                    "infrared2": f"{OUTPUT_DIR}/ir2_{tsname}.avi",
                    "depth":     f"{OUTPUT_DIR}/depth_{tsname}.avi",
                    "web":       f"{OUTPUT_DIR}/web_{tsname}.avi",
                    "rs_color":  f"{OUTPUT_DIR}/rs_color_{tsname}.avi"  # [NEW]
                })
                recording = True
                print("[main] REC started")
            else:
                write_q.put({"cmd": "stop"})
                t0 = time.time()
                while time.time() - t0 < 0.45:
                    time.sleep(0.05)
                try:
                    pipeline.stop()
                except Exception:
                    pass
                time.sleep(0.45)
                try:
                    pipeline.start(config)
                except Exception as e:
                    print("[main] failed restart pipeline:", e)
                recording = False
                print("[main] REC stopped")

        if time.time() - last_fps >= 5:
            if len(fps_ts) > 1:
                fps = (len(fps_ts)-1)/(fps_ts[-1]-fps_ts[0])
                print(f"[fps] {fps:.2f}, fq={frame_q.qsize()}, wq={write_q.qsize()}")
            last_fps = time.time()

    write_q.put({"cmd": "exit"})
    write_q.put(None)
    cv2.destroyAllWindows()
    stop_event.set()
    print("exiting...")

if __name__ == "__main__":
    main()