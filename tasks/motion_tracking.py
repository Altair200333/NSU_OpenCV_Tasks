import cv2 as cv
import numpy as np

from tools import *

img = cv.imread("../imgs/lines/check.jpg")
img = clipImg(img, 600)

canvas = np.zeros(img.shape, dtype=np.uint8)
while True:

    canvas[:, :, :] = 0
    input = np.float32([[0, 0], [img.shape[1] - 1, 0], [img.shape[1] - 1, img.shape[0] - 1], [0, img.shape[0] - 1]])
    output = np.float32([[100,100], [500, 100], [400, 400], [200, 400]])
    matrix = cv.getPerspectiveTransform(input, output)

    imgOutput = cv.warpPerspective(img, matrix, img.shape[:2], cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT,borderValue=(0, 0, 0))
    cv.imshow("warp", imgOutput)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(gray, 100, 0.03, 10)
    corners_int = np.int0(corners)

    for corner in corners_int:
        x, y = corner.ravel()
        cv.drawMarker(canvas, (x, y), (255, 100, 0))

    cv.imshow('img', cv.add(img, canvas))

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
