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
    # 1. Threshold filter (optional)
    depth = th_filter.process(depth)

    # 2. Decimation (optional, keep original resolution if magnitude = 1)
    depth = dec_filter.process(depth)

    # 3. Convert depth → disparity
    depth = to_disparity.process(depth)

    # 4. Spatial filtering (edge-preserving smoothing)
    depth = spa_filter.process(depth)

    # 5. Temporal filtering (noise reduction across frames)
    depth = tmp_filter.process(depth)

    # 6. Convert disparity → depth
    depth = to_depth.process(depth)

    # 7. Hole filling (run last)
    depth = hole_filter.process(depth)

    return depth

def start_realsense_with_recording(bag_path):
    global pipeline, config
    try:
        pipeline.stop()
    except Exception:
        pass
    time.sleep(0.35)   # allow previous bag to finalize

    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)
    cfg.enable_stream(rs.stream.infrared, 2, WIDTH, HEIGHT, rs.format.y8, FPS)
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
            depth = frames.get_depth_frame()
            ir1 = frames.get_infrared_frame(1)
            ir2 = frames.get_infrared_frame(2)

            if depth and ir1 and ir2:
                depth = apply_depth_filters(depth)

                depth_np = np.asanyarray(depth.get_data())
                ir1_np = np.asanyarray(ir1.get_data())
                ir2_np = np.asanyarray(ir2.get_data())

                frame_q.put((time.time(), ir1_np, ir2_np, depth_np), block=False)
        except queue.Full:
            # drop frame if queue is full
            pass
        except Exception:
            # avoid killing grabber on transient errors
            pass

    try:
        pipeline.stop()
    except Exception:
        pass
    stop_event.set()
    print("[grabber] exited")

# ---------- Writer ----------
def writer():
    wr_ir1 = wr_ir2 = wr_depth = wr_web = None
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
                fourcc = cv2.VideoWriter_fourcc(*COLOR_CODEC)
                wr_ir1   = cv2.VideoWriter(p_ir1,  fourcc, FPS, (WIDTH, HEIGHT))
                wr_ir2   = cv2.VideoWriter(p_ir2,  fourcc, FPS, (WIDTH, HEIGHT))
                wr_depth = cv2.VideoWriter(p_d,    fourcc, FPS, (WIDTH, HEIGHT))
                wr_web   = cv2.VideoWriter(p_w,    fourcc, FPS, (WIDTH, HEIGHT))
                if not (wr_ir1.isOpened() and wr_ir2.isOpened() and wr_depth.isOpened() and wr_web.isOpened()):
                    print("[writer] failed to open writers")
                    # release any opened
                    if wr_ir1: wr_ir1.release()
                    if wr_ir2: wr_ir2.release()
                    if wr_depth: wr_depth.release()
                    if wr_web: wr_web.release()
                    recording = False
                else:
                    recording = True
                    print("[writer] recording ->", p_ir1, p_ir2, p_d, p_w)

            elif cmd == "frame" and recording:
                # expect BGR frames already
                wr_ir1.write(item["infrared1"])
                wr_ir2.write(item["infrared2"])
                wr_depth.write(item["depth"])
                wr_web.write(item["web"])

            elif cmd == "stop" and recording:
                if wr_ir1: wr_ir1.release()
                if wr_ir2: wr_ir2.release()
                if wr_depth: wr_depth.release()
                if wr_web: wr_web.release()
                recording = False
                print("[writer] stopped")

            elif cmd == "exit":
                if wr_ir1: wr_ir1.release()
                if wr_ir2: wr_ir2.release()
                if wr_depth: wr_depth.release()
                if wr_web: wr_web.release()
                print("[writer] exiting")
                break
        except Exception as e:
            print("[writer] exception:", e)
            # continue loop (do not kill writer thread)

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

    t_grab = threading.Thread(target=grabber, daemon=True)
    t_write = threading.Thread(target=writer, daemon=True)
    t_grab.start()
    t_write.start()

    recording = False
    last_fps = time.time()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    while not stop_event.is_set():
        try:
            ts, ir1, ir2, depth = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue

        fps_ts.append(time.time())
        ret, web = cam.read()
        if not ret:
            web = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        dcolor = depth_to_color(depth)

        # Show IR as grayscale (convert to 3-channel for drawing REC mark)
        vis_ir1 = cv2.cvtColor(ir1, cv2.COLOR_GRAY2BGR)
        vis_ir2 = cv2.cvtColor(ir2, cv2.COLOR_GRAY2BGR)

        if recording:
            # draw indicator on the visible copy, not the raw arrays (we queue BGR)
            cv2.circle(vis_ir1, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(vis_ir1, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # show windows
        cv2.imshow("RealSense IR1", vis_ir1)
        cv2.imshow("RealSense IR2", vis_ir2)
        cv2.imshow("RealSense Depth Performance", dcolor)
        cv2.imshow("Webcam", web)

        # when recording, queue BGR frames for writer (writer expects BGR)
        if recording:
            try:
                write_q.put_nowait({
                    "cmd": "frame",
                    "infrared1": vis_ir1.copy(),
                    "infrared2": vis_ir2.copy(),
                    "depth": dcolor.copy(),
                    "web": web.copy()
                })
            except queue.Full:
                print("[main] write_q full, dropping frame")

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            stop_event.set()
            break

        elif k == ord('r'):
            if not recording:
                tsname = time.strftime("%Y%m%d-%H%M%S")
                bag_path = os.path.join(OUTPUT_DIR, f"session_{tsname}.bag")

                # restart pipeline to start bag recording (safe)
                start_realsense_with_recording(bag_path)

                # start avi writers
                write_q.put({
                    "cmd": "start",
                    "infrared1": f"{OUTPUT_DIR}/ir1_{tsname}.avi",
                    "infrared2": f"{OUTPUT_DIR}/ir2_{tsname}.avi",
                    "depth": f"{OUTPUT_DIR}/depth_{tsname}.avi",
                    "web": f"{OUTPUT_DIR}/web_{tsname}.avi"
                })
                recording = True
                print("[main] REC started")
            else:
                # stop avi writers first
                write_q.put({"cmd": "stop"})
                # wait short time for writer to release files
                time_wait = 0.45
                t0 = time.time()
                while time.time() - t0 < time_wait:
                    # give writer a chance to flush; small sleep avoids busy wait
                    time.sleep(0.05)

                # now stop pipeline to finalize bag (then restart normal streaming)
                try:
                    pipeline.stop()
                except Exception:
                    pass
                time.sleep(0.45)
                # restart with normal (non-recording) config
                try:
                    pipeline.start(config)
                except Exception as e:
                    print("[main] failed restart pipeline:", e)

                recording = False
                print("[main] REC stopped")

        # fps log
        if time.time() - last_fps >= 5:
            if len(fps_ts) > 1:
                fps = (len(fps_ts)-1)/(fps_ts[-1]-fps_ts[0])
                print(f"[fps] {fps:.2f}, fq={frame_q.qsize()}, wq={write_q.qsize()}")
            last_fps = time.time()

    # shutdown
    write_q.put({"cmd": "exit"})
    write_q.put(None)
    cv2.destroyAllWindows()
    stop_event.set()
    print("exiting...")

if __name__ == "__main__":
    main()
