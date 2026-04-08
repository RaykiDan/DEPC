from __future__ import annotations

import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2


# ── Encoder backbone configurations ──────────────────────────────────────────
#
#   These are the same for both relative and metric models.
#   Size/speed tradeoff:
#     vits → smallest, fastest, least accurate
#     vitb → medium
#     vitl → largest,  slowest, most accurate
#
ENCODER_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,   96,   192,  384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,   192,  384,  768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256,  512,  1024, 1024]},
}

# ── Checkpoint paths ──────────────────────────────────────────────────────────

CHECKPOINT_PATHS = {
    "relative": {
        "vits": "checkpoints/depth_anything_v2_vits.pth",
        "vitb": "checkpoints/depth_anything_v2_vitb.pth",
        "vitl": "checkpoints/depth_anything_v2_vitl.pth",
    },
    "metric": {
        "vits": "checkpoints/depth_anything_v2_metric_hypersim_vits.pth",
        "vitb": "checkpoints/depth_anything_v2_metric_hypersim_vitb.pth",
        "vitl": "checkpoints/depth_anything_v2_metric_hypersim_vitl.pth",
    },
}

# ── Metric model extra parameter ─────────────────────────────────────────────
#
#   The metric model constructor requires max_depth, which the relative
#   model does NOT have. Passing it to the wrong model causes a TypeError.
#
#     hypersim (indoor)  → max_depth = 20  metres
#     vkitti   (outdoor) → max_depth = 80  metres
#
METRIC_MAX_DEPTH = 20   # indoor / hypersim


class DepthModel:
    """
    Wraps DepthAnythingV2 with lazy loading and encoder/mode switching.

    Usage
    -----
        model = DepthModel(encoder="vits", mode="metric")
        depth_array = model.infer(rgb_frame)
        # → H×W float32 in metres       (metric mode)
        # → H×W float32 unitless        (relative mode, larger = closer)
    """

    def __init__(self, encoder: str = "vits", mode: str = "metric"):
        """
        Parameters
        ----------
        encoder : "vits" | "vitb" | "vitl"
        mode    : "relative" | "metric"
        """
        self.encoder = encoder
        self.mode    = mode
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model          = None
        self._loaded_encoder = None
        self._loaded_mode    = None

        print(f"[DepthModel] Initialized — encoder={encoder}, mode={mode}, device={self.device}")

    # ── Public API ────────────────────────────────────────────────────────────

    def set_encoder(self, encoder: str):
        """Switch encoder. Model reloads on the next infer() call."""
        if encoder not in ENCODER_CONFIGS:
            print(f"[DepthModel] Unknown encoder '{encoder}'. Valid: {list(ENCODER_CONFIGS)}")
            return
        self.encoder = encoder

    def set_mode(self, mode: str):
        """Switch between 'relative' and 'metric'. Model reloads on next infer() call."""
        if mode not in ("relative", "metric"):
            print(f"[DepthModel] Unknown mode '{mode}'. Use 'relative' or 'metric'.")
            return
        self.mode = mode

    def infer(self, rgb_frame: np.ndarray) -> np.ndarray | None:
        """
        Run depth estimation on a single RGB frame.

        Parameters
        ----------
        rgb_frame : H×W×3 uint8 NumPy array in RGB order.

        Returns
        -------
        H×W float32 NumPy array, or None on failure.
        """
        if not self._ensure_loaded():
            return None

        try:
            raw = self._model.infer_image(rgb_frame)
            return raw.astype(np.float32)
        except Exception as e:
            print(f"[DepthModel] Inference failed: {e}")
            return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> bool:
        """Load (or reload) the model only if encoder or mode changed."""
        already_loaded = (
            self._model is not None
            and self._loaded_encoder == self.encoder
            and self._loaded_mode    == self.mode
        )
        if already_loaded:
            return True

        checkpoint = CHECKPOINT_PATHS.get(self.mode, {}).get(self.encoder)
        if checkpoint is None:
            print(f"[DepthModel] No checkpoint for mode='{self.mode}', encoder='{self.encoder}'")
            return False

        print(f"[DepthModel] Loading — encoder={self.encoder}, mode={self.mode} ...")
        try:
            config = ENCODER_CONFIGS[self.encoder].copy()
            if self.mode == "metric":
                config["max_depth"] = METRIC_MAX_DEPTH

            model = DepthAnythingV2(**config)

            state_dict = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state_dict)
            model = model.to(self.device).eval()

            self._model          = model
            self._loaded_encoder = self.encoder
            self._loaded_mode    = self.mode

            print(f"[DepthModel] Loaded successfully.")
            return True

        except Exception as e:
            print(f"[DepthModel] Failed to load: {e}")
            self._model = None
            return False