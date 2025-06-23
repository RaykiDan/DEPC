import cv2
import numpy as np

# Buka dua video IR (left dan right)
cap_left = cv2.VideoCapture("records/cam2.avi")   # left
cap_right = cv2.VideoCapture("records/cam3.avi")  # right

# Buat objek StereoSGBM
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*6,  # harus kelipatan 16
    blockSize=7,
    P1=8*3*7**2,
    P2=32*3*7**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

while cap_left.isOpened() and cap_right.isOpened():
    retL, frameL = cap_left.read()
    retR, frameR = cap_right.read()
    if not retL or not retR:
        break

    # Konversi ke grayscale (jika belum)
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Hitung disparity map
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Normalisasi ke 0–255 untuk ditampilkan
    disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_uint8 = np.uint8(disp_norm)

    # Ubah ke colormap
    disp_colormap = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_JET)

    # Tampilkan
    cv2.imshow("Disparity Colormap", disp_colormap)
    if cv2.waitKey(1) == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
