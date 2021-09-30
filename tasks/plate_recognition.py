import cv2 as cv
import numpy as np
from pytesseract import pytesseract

from image_transform import *
from tasks.tracking_tools import *
from tools import *
from color_correction import *
import imutils

img = cv.imread("../imgs/number_plates/car1.jpg")
img = clipImg(img, 1000)


def approx_contour(cntr):
    perimeter = cv.arcLength(cntr, True)
    approx = cv.approxPolyDP(cntr, 0.01 * perimeter, True)
    return approx


def filter_contours(cntrs):
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
    'tresh': 255.0,
    'max_val': 255.0,
}


def set_param(name, x):
    global params
    params[name] = x


controls_window = 'controls'
cv.namedWindow(controls_window)

cv.createTrackbar('tresh_1', controls_window, 0, 255, lambda x: set_param('tresh_1', x))
cv.createTrackbar('tresh_2', controls_window, 255, 255, lambda x: set_param('tresh_2', x))

cv.createTrackbar('tresh', controls_window, 50, 255, lambda x: set_param('tresh', x))
cv.createTrackbar('max_val', controls_window, 255, 255, lambda x: set_param('max_val', x))


def parseText(screenCnt):
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
    thresh = cv.adaptiveThreshold(Cropped, params['max_val'], cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, params['tresh'])
    text = pytesseract.image_to_string(thresh, lang='eng',
                                       config='--oem 3 -l eng --psm 6')
    print(text)
    cv.imshow('new', thresh)


while True:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.blur(gray, (3, 3))
    # thresh = cv.adaptiveThreshold(gray, params['max_val'], cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 3, params['tresh'])

    edged = cv.Canny(gray, params['tresh_1'], params['tresh_2'])

    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = filter_contours(contours)

    canvas = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    cv.drawContours(canvas, contours, -1, (255, 30, 0), 2)
    parseText(contours[1])
    # for cntr in contours:
    #    parseText(cntr)
    # cv.imshow("tresh", gray)
    cv.imshow("img", canvas)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
