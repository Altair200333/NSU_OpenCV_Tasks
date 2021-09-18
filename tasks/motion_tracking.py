import cv2 as cv
import numpy as np

from tools import *

img = cv.imread("../imgs/lines/check.jpg")
img = clipImg(img, 600)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray, 100, 0.03, 10)
corners_int = np.int0(corners)

canvas = np.zeros(img.shape, dtype=np.uint8)
while True:

    canvas[:, :, :] = 0
    for corner in corners_int:
        x, y = corner.ravel()
        cv.drawMarker(canvas, (x, y), (255, 100, 0))

    cv.imshow('img', cv.add(img, canvas))

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
