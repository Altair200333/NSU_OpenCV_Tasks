import cv2 as cv
import math
import numpy as np
from tools import *


img = cv.imread("../imgs/car.jpg")
img = scaleImg(img, 0.5)

hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)

low = np.array((0, 100, 100), np.uint8)
high = np.array((255, 150, 150), np.uint8)

selectedColor = np.array((250, 250, 255), np.uint8)

drawing = False
radius = 10


def line_drawing(event, x, y, flags, param):
    global selectedColor, low, high, drawing, radius
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv.EVENT_MOUSEMOVE and drawing:
        selectedColor = lab_img[y, x, :]
        updateFilter(radius, selectedColor)

    if event == cv.EVENT_MOUSEWHEEL:
        if flags < 0:
            radius -= 1
        else:
            radius += 1
        updateFilter(radius, selectedColor)


def updateFilter(radius, selectedColor):
    global low, high
    low = np.array((0, np.clip(selectedColor[1] - radius, 0, 255), np.clip(selectedColor[2] - radius, 0, 255)),
                   np.uint8)
    high = np.array((255, np.clip(selectedColor[1] + radius, 0, 255), np.clip(selectedColor[2] + radius, 0, 255)),
                    np.uint8)


cv.imshow("img", img)

cv.setMouseCallback('img', line_drawing)

while (1):
    mask = cv.inRange(lab_img, low, high)

    result = cv.bitwise_and(img, img, mask=mask)
    cv.imshow("mask", mask)
    cv.imshow("result", result)

    lab_copy = lab_img.copy()
    lab_copy[:, :, 0] = 180
    cv.imshow("lab", cv.cvtColor(lab_copy, cv.COLOR_Lab2BGR))
    cv.setMouseCallback('lab', line_drawing)

    k = cv.waitKey(2) & 0xFF
    if k == 27:
        break
