import cv2 as cv
import numpy as np
from tools import *
from imutils import contours

img = cv.imread('../imgs/number_plates/car1.jpg')
img = clipImg(img, 600)

controlls_window_name = 'controlls'
cv.namedWindow(controlls_window_name)


def pass_callback(x):
    pass


treshold1 = 100
treshold2 = 100


def set_treshold1(x):
    global treshold1
    treshold1 = x


def set_treshold2(x):
    global treshold2
    treshold2 = x


mode = 0
modes_count = 2


def nextMode(x):
    global mode
    mode = (mode + x) % modes_count


cv.createTrackbar('treshold 1', controlls_window_name, 200, 500, set_treshold1)
cv.createTrackbar('treshold 2', controlls_window_name, 220, 500, set_treshold2)

canvas = np.zeros(img.shape, dtype=np.uint8)
overlay = np.zeros(img.shape, dtype=np.uint8)

while True:
    edges_canny = cv.Canny(img, treshold1, treshold2)

    contours, hierarchy = cv.findContours(edges_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    canvas[:, :, :] = 0
    overlay[:, :, :] = 0
    for c in contours:
        # Obtain bounding rectangle for each contour
        x, y, w, h = cv.boundingRect(c)

        # Draw bounding box rectangle, crop using Numpy slicing
        cv.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv.drawContours(overlay, contours, -1, (0, 255, 0))

    cv.imshow('img', cv.add(img, overlay))

    result = canvas
    if mode == 0:
        result = cv.add(cv.cvtColor(edges_canny, cv.COLOR_GRAY2BGR), canvas)
    else:
        result = cv.cvtColor(edges_canny, cv.COLOR_GRAY2BGR)

    cv.imshow("edges", result)

    k = cv.waitKey(1) & 0xFF

    if k == ord('q'):
        nextMode(-1)
    if k == ord('e'):
        nextMode(1)
    if k == 27:
        break
