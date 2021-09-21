import cv2 as cv
import numpy as np

from image_transform import *
from tools import *
from color_correction import *
from tasks.tracking_tools import *
from scipy.ndimage import label

frame_size = 600

img = cv.imread("../imgs/coins2.jpg")
img = clipImg(img, frame_size)

controls_window = 'controls'
cv.namedWindow(controls_window)

params = {
    'iterations': 5,
    'tresh_1': 0,
    'tresh_2': 255
}


def set_param(name, x):
    global params
    params[name] = x


cv.createTrackbar('iterations', controls_window, 5, 15, lambda x: set_param('iterations', x))
cv.createTrackbar('tresh_1', controls_window, 0, 255, lambda x: set_param('tresh_1', x))
cv.createTrackbar('tresh_2', controls_window, 255, 255, lambda x: set_param('tresh_2', x))


def segment_on_dt(a, img):
    global params
    border = cv.dilate(img, None, iterations=params['iterations'])
    border = border - cv.erode(border, None)

    dt = cv.distanceTransform(img, cv.DIST_L2, 3)
    cv.imshow('distances', dt)

    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv.threshold(dt, params['tresh_1'], params['tresh_2'], cv.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))

    # Completing the markers now.
    lbl[border == 255] = 255

    lbl = lbl.astype(np.int32)
    labels = cv.watershed(a, lbl)

    cv.imshow('labels', cv.convertScaleAbs(labels))

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl


img_copy = img.copy()
while True:
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img_bin = cv.threshold(img_gray, 0, 255, cv.THRESH_OTSU)

    img_bin = cv.morphologyEx(img_bin, cv.MORPH_OPEN, np.ones((3, 3), dtype=int))
    result = segment_on_dt(img, img_bin)

    result[result != 255] = 0
    result = cv.dilate(result, None)

    img_copy[result == 255] = (200, 100, 15)

    cv.imshow('outline', result)
    cv.imshow('edges', img_copy)
    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
