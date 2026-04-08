import math
import numpy as np
import pyrealsense2 as rs


# Filter chain configuration
# (The filter order matters. This is the recommended Intel pipeline)
#
#   1. Threshold    → discard pixels outside [min_dist, max_dist]
#   2. Decimation   → reduce resolution (speeds up everything downstream)
#   3. To Disparity → convert depth → disparity space (required before spatial/temporal)
#   4. Spatial      → smooth across neighbouring pixels (fills small holes)
#   5. Temporal     → smooth across frames (reduces flicker)
#   6. To Depth     → convert back from disparity → depth space
#   7. Hole Filling → fill remaining holes with neighbouring valid depth

DEPTH_MIN_M = 0.2   # metres — pixels closer than this are discarded
DEPTH_MAX_M = 4.0   # metres — pixels farther than this are discarded


class RealSenseReader:
    """
    Wraps an Intel RealSense .bag file for frame-by-frame depth reading.

    Usage
    -----
        reader = RealSenseReader()
        reader.open("path/to/recorded.bag")

        depth_metres = reader.read()    # H×W float32 NumPy array, or None

        fov = reader.get_fov()          # {"ir1": (h, v), "ir2": (h, v)} in degrees

        reader.close()
    """

    def __init__(self):
        self._pipeline = None
        self._profile  = None
        self._align    = rs.align(rs.stream.infrared)

        # ── Build filter objects once, reuse every frame ──────────────────
        self._th_filter  = rs.threshold_filter()
        self._th_filter.set_option(rs.option.min_distance, DEPTH_MIN_M)
        self._th_filter.set_option(rs.option.max_distance, DEPTH_MAX_M)

        self._dec_filter  = rs.decimation_filter()
        self._spa_filter  = rs.spatial_filter()
        self._tmp_filter  = rs.temporal_filter()
        self._hole_filter = rs.hole_filling_filter()
        self._to_disp     = rs.disparity_transform(True)   # depth  → disparity
        self._to_depth    = rs.disparity_transform(False)  # disparity → depth

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self, bag_path: str) -> bool:
        try:
            config = rs.config()
            config.enable_device_from_file(bag_path, repeat_playback=True)

            self._pipeline = rs.pipeline()
            self._profile  = self._pipeline.start(config)

            # Disable real-time playback so we never drop frames
            playback = self._profile.get_device().as_playback()
            playback.set_real_time(False)

            print(f"[RealSenseReader] Opened: {bag_path}")
            return True

        except Exception as e:
            print(f"[RealSenseReader] Failed to open '{bag_path}': {e}")
            self._pipeline = None
            self._profile  = None
            return False

    def close(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
            self._profile  = None
            print("[RealSenseReader] Closed.")

    @property
    def is_open(self) -> bool:
        return self._pipeline is not None

    # ── Frame reading ─────────────────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        if not self.is_open:
            return None

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=50)
            frames = self._align.process(frames)

            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None

            filtered    = self._apply_filters(depth_frame)
            depth_raw   = np.asanyarray(filtered.get_data()).astype(np.float32)
            depth_m     = depth_raw * 0.001     # millimetres → metres
            return depth_m

        except Exception as e:
            print(f"[RealSenseReader] Failed to read frame: {e}")
            return None

    # ── FOV ───────────────────────────────────────────────────────────────────

    def get_fov(self) -> dict | None:
        """
        Calculate the horizontal and vertical FOV for both IR cameras.

        Returns
        -------
        {
            "ir1": (fov_h_degrees, fov_v_degrees),
            "ir2": (fov_h_degrees, fov_v_degrees),
        }
        or None if the pipeline is not open.
        """
        if self._profile is None:
            return None

        try:
            result = {}
            for key, index in (("ir1", 1), ("ir2", 2)):
                stream     = self._profile.get_stream(rs.stream.infrared, index)
                intrinsics = stream.as_video_stream_profile().get_intrinsics()
                result[key] = self._fov_from_intrinsics(intrinsics)
            return result

        except Exception as e:
            print(f"[RealSenseReader] Failed to get FOV: {e}")
            return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_filters(self, depth_frame) -> rs.depth_frame:
        """Run the full filter chain on a raw depth frame."""
        frame = self._th_filter.process(depth_frame)
        frame = self._dec_filter.process(frame)
        frame = self._to_disp.process(frame)
        frame = self._spa_filter.process(frame)
        frame = self._tmp_filter.process(frame)
        frame = self._to_depth.process(frame)
        frame = self._hole_filter.process(frame)
        return frame.as_depth_frame()

    @staticmethod
    def _fov_from_intrinsics(intrinsics) -> tuple[float, float]:
        """Convert camera intrinsics to (fov_h, fov_v) in degrees."""
        fov_h = 2 * math.atan(intrinsics.width  / (2 * intrinsics.fx)) * (180 / math.pi)
        fov_v = 2 * math.atan(intrinsics.height / (2 * intrinsics.fy)) * (180 / math.pi)
        return round(fov_h, 2), round(fov_v, 2)