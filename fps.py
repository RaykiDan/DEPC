# Program only to test FPS and print the overall result

import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

start_time = time.time()
frame_count = 0

while time.time() - start_time < 5:
    ret, frame = cap.read()
    if ret:
        frame_count += 1

cap.release()
print("Measured FPS:", frame_count / 5)
