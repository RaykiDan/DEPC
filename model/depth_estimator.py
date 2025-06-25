# depth_backend.py

import os
import numpy as np
import torch
import cv2
import time
import matplotlib.cm as cm
from depth_anything_v2.dpt import DepthAnythingV2

class DepthEstimator:
    def __init__(self, encoder='vits', device=None):
        self.encoder = encoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()

    def load_model(self):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
        }
        config = model_configs[self.encoder]
        model = DepthAnythingV2(**config)

        ckpt_path = f'checkpoints/depth_anything_v2_{self.encoder}.pth'
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model = model.to(self.device).eval()
        return model

    def estimate_depth(self, frame: np.ndarray, input_size=518) -> np.ndarray:
        start = time.time()
        with torch.no_grad():
            depth = self.model.infer_image(frame, input_size=input_size)
        print("Waktu infer per frame:", time.time() - start, "detik")

        """Input: BGR image (np.ndarray), Output: colored depth image"""
        with torch.no_grad():
            depth = self.model.infer_image(frame, input_size=input_size)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)

            # Apply colormap
            cmap = cm.get_cmap('Spectral_r')
            depth_color = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
            depth_color = depth_color[:, :, ::-1]  # convert RGB to BGR for OpenCV

            return depth_color