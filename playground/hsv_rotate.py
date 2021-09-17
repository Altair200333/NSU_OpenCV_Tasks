import cv2 as cv
import numpy as np
from tools import *

img = cv.imread('../imgs/portraits/niki.jpg')
img = clipImg(img, 600)

gradient_map = np.ones((256, 256, 3), np.uint8)
for i in range(0, 255):
    for j in range(0, 255):
        gradient_map[i, j] = [200, j, 255 - i]

angle = 0


def angle_changed(x):
    global angle
    angle = x


hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

cv.imshow("grad", cv.cvtColor(gradient_map, cv.COLOR_Lab2BGR))
cv.namedWindow("img")
cv.createTrackbar('rotation', 'img', 0, 360, angle_changed)

while True:

    hsv_mod = hsv_img.copy()

    hsv_mod[:, :, 0] = (hsv_mod[:, :, 0] + angle) % 180
    cv.imshow("img", cv.cvtColor(hsv_mod, cv.COLOR_HSV2BGR))

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
