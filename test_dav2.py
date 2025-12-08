import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("[INFO] Device:", device)

# ====== LOAD MODEL ======
model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
state = torch.load("checkpoints/depth_anything_v2_metric_hypersim_vits.pth", map_location='cpu')
model.load_state_dict(state)
model = model.to(device).eval()

# ====== LOAD TEST IMAGE ======
img = cv2.imread("demo10.jpg")   # pakai gambar apapun
if img is None:
    print("Gambar tidak ditemukan!")
    exit()

# ====== PREPROCESS ======
size = 252
resized = cv2.resize(img, (size, size))
rgb = resized[:, :, ::-1] / 255.0
rgb = (rgb - 0.5) / 0.5
ten = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

# ====== INFERENCE ======
with torch.no_grad():
    out = model(ten)

raw = out[0].cpu().numpy().squeeze()

print("Raw depth min/max:", raw.min(), raw.max())

dn = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
dm = (dn * 255).astype(np.uint8)
dm_color = cv2.applyColorMap(dm, cv2.COLORMAP_JET)

cv2.imshow("input", img)
cv2.imshow("dav2 depth", dm_color)
cv2.waitKey(0)
