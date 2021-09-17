import cv2 as cv
import numpy as np
from tools import *

img = cv.imread("imgs/page2.png")

img = clipImg(img, 600)

treshVal = 0
blocksizeVal = 21


def tresh(x):
    global treshVal
    treshVal = x


def blocksize(x):
    global blocksizeVal
    if x < 1:
        x += 1
    if x % 2 == 0:
        x += 1
    blocksizeVal = x


gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.namedWindow('img')
cv.createTrackbar('tresh', 'img', 0, 100, tresh)
cv.createTrackbar('block', 'img', 11, 49, blocksize)

while True:
    blured = gris  # cv.GaussianBlur(gris, (3, 3), cv.BORDER_DEFAULT)
    mask = cv.adaptiveThreshold(blured, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 221,
                                treshVal)
    mask = cv.bitwise_not(mask)
    cv.imshow("mask", mask)
    cv.imshow("blured", blured)

    cv.imshow("img", img)
    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
