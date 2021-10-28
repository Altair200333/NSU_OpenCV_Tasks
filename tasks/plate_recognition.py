import cv2 as cv
import numpy as np
from pytesseract import pytesseract

from image_transform import *
from tasks.tracking_tools import *
from tools import *
from color_correction import *
import imutils
from image_transform import *
import statistics
import re
import easyocr
from alpr_tools import *

img = cv.imread("../imgs/number_plates/car1.jpg")
img = clipImg(img, 700)

reader = easyocr.Reader(['en'], gpu=True)


def approx_contour(cntr):
    perimeter = cv.arcLength(cntr, True)
    approx = cv.approxPolyDP(cntr, 0.02 * perimeter, True)
    return approx


def filter_rectangular_contours(cntrs):
    filtered = []
    for c in cntrs:
        approx = approx_contour(c)

        ref = cv.contourArea(c)
        tar = cv.contourArea(approx)
        ar = 0 if tar == 0 else ref / tar

        if len(approx) == 4 and ref > 50 and 0.5 <= ar <= 1.5:
            filtered.append(approx)
    return filtered


params = {
    'tresh_1': 0.0,
    'tresh_2': 255.0,
    'bloor': 1,
}


def set_param(name, x):
    global params
    params[name] = x


controls_window = 'controls'
cv.namedWindow(controls_window)

cv.createTrackbar('tresh_1', controls_window, 0, 255, lambda x: set_param('tresh_1', x))
cv.createTrackbar('tresh_2', controls_window, 255, 255, lambda x: set_param('tresh_2', x))

cv.createTrackbar('bloor', controls_window, 1, 1, lambda x: set_param('bloor', x))

while True:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if params['bloor'] == 1:
        gray = cv.blur(gray, (3, 3))

    canvas = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    edged = cv.Canny(gray, params['tresh_1'], params['tresh_2'])

    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = filter_rectangular_contours(contours)

    for cnt in contours:
        res = process_rectangle(reader, gray, cnt)

        if res is not None:
            (bbox, text, prob) = res
            # unpack the bounding box
            x, y, w, h = cv.boundingRect(cnt)
            # cleanup the text and draw the box surrounding the text along
            # with the OCR'd text itself
            cv.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(canvas, text, (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # what we thought to be a number plate
    cv.drawContours(canvas, contours, -1, (255, 30, 0), 2)
    cv.imshow("img", canvas)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
