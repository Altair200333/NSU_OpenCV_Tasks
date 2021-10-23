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

img = cv.imread("../imgs/number_plates/car1.jpg")
img = clipImg(img, 700)


def approx_contour(cntr):
    perimeter = cv.arcLength(cntr, True)
    approx = cv.approxPolyDP(cntr, 0.02 * perimeter, True)
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
    thresh = cv.adaptiveThreshold(Cropped, params['max_val'], cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5,
                                  params['tresh'])
    text = pytesseract.image_to_string(thresh, lang='eng',
                                       config='--oem 3 -l eng --psm 6')
    print(text)
    cv.imshow('new', thresh)


def process_rectangle(img, cntr):
    x, y, w, h = cv.boundingRect(cntr)
    res = np.zeros((h, w), dtype=np.uint8)

    res = warpImageFit(img, cntr[0][0], cntr[1][0], cntr[2][0], cntr[3][0], res)

    # fix flipness
    res = np.fliplr(res)

    inverted = preprocess_plate(res)

    contours = find_cntrs(inverted)

    # sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0])

    canvas = cv.cvtColor(inverted, cv.COLOR_GRAY2BGR)
    # cv.drawContours(canvas, sorted_contours, -1, (255, 30, 0), 2)

    filtered_cntrs = []
    for letter in sorted_contours:
        x, y, w, h = cv.boundingRect(letter)
        if 0.5 <= h / canvas.shape[0] <= 1 and 0.05 <= w / canvas.shape[1] <= 0.2:
            filtered_cntrs.append(letter)

    found_text = ""
    for letter in filtered_cntrs:
        x, y, w, h = cv.boundingRect(letter)
        roi = inverted[y - 10:y + h + 10, x - 10:x + w + 10]

        #cv.imshow(str(x), roi)
        text = pytesseract.image_to_string(roi,
                                           config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')

        clean_text = re.sub('[\W_]+', '', text)
        found_text += clean_text

        cv.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(found_text)
    cv.imshow('region', canvas)


def find_cntrs(inverted):
    try:
        contours, hierarchy = cv.findContours(inverted, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv.findContours(inverted, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours


def preprocess_plate(res):
    # resize image to three times as large as original for better readability
    gray = cv.resize(res, None, fx=3, fy=3, interpolation=cv.INTER_CUBIC)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
    # create rectangular kernel for dilation
    rect_kern = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # apply dilation to make regions more clear
    dilation = cv.dilate(thresh, rect_kern, iterations=1)
    inverted = 255 - dilation
    return inverted


while True:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.blur(gray, (3, 3))
    # thresh = cv.adaptiveThreshold(gray, params['max_val'], cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 3, params['tresh'])

    edged = cv.Canny(gray, params['tresh_1'], params['tresh_2'])

    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = filter_contours(contours)

    for cnt in contours:
        process_rectangle(gray, cnt)

    canvas = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    cv.drawContours(canvas, contours, -1, (255, 30, 0), 2)
    cv.imshow("img", canvas)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
