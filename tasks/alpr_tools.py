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


def process_rectangle(reader, img, cntr):
    x, y, w, h = cv.boundingRect(cntr)
    res = np.zeros((h, w), dtype=np.uint8)

    res = warpImageFit(img, cntr[0][0], cntr[1][0], cntr[2][0], cntr[3][0], res)

    # fix flipness
    res = np.fliplr(res)

    inverted = preprocess_plate(res)

    res = reader.readtext(inverted)
    if len(res) == 0:
        return None

    (bbox, text, prob) = res[0]
    text = re.sub('[\W_]+', '', text)

    # print(text)
    # cv.imshow('region', inverted)

    return bbox, text, prob
