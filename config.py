# ── config.py ─────────────────────────────────────────────────────────────────
#
#   Single source of truth for all shared settings across the project.
#   Change values here — they will automatically apply everywhere.
#
# ─────────────────────────────────────────────────────────────────────────────


# ── Depth display range (metres) ──────────────────────────────────────────────
#
#   Controls the colormap scale on all depth frames and rulers.
#   Pixels closer than DMIN → clamped to the near color (blue end).
#   Pixels farther than DMAX → clamped to the far color  (red end).
#
DMIN = 0.5   # metres
DMAX = 4.0   # metres

# ── Active model settings ─────────────────────────────────────────────────────
#
#   These are updated at runtime by SettingWindow when the user hits Apply.
#   They are read back by SettingWindow on open to reflect the current state.
#
CURRENT_ENCODER = "vits"     # "vits" | "vitb" | "vitl"
CURRENT_MODE    = "metric"   # "metric" | "relative"


# ── Webcam FOV (degrees) ──────────────────────────────────────────────────────
#
#   Fixed hardware spec for the RGB webcam.
#   Only used for the FOV label display — does not affect depth estimation.
#
WEBCAM_FOV_H = 64.26
WEBCAM_FOV_V = 50.35


# ── Object annotations for the horizontal ruler ───────────────────────────────
#
#   Each entry marks a known object's depth range on the ruler bar.
#   "color" is BGR (OpenCV convention).
#
ANNOTATIONS = [
    {"name": "Kardus",      "depth_min": 1.6, "depth_max": 1.9, "color": (0, 0, 0)},
    {"name": "Kursi",       "depth_min": 1.9, "depth_max": 2.2, "color": (0, 0, 0)},
    {"name": "Papan Tulis", "depth_min": 2.2, "depth_max": 2.7, "color": (0, 0, 0)},
]


# ── RealSense filter range (metres) ───────────────────────────────────────────
#
#   Pixels outside [RS_DMIN, RS_DMAX] are discarded by the threshold filter
#   before any other processing. Can be wider than DMIN/DMAX above since
#   the colormap clamps the display range independently.
#
RS_DMIN = 0.2
RS_DMAX = 3.0