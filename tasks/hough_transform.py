import cv2 as cv
import numpy as np

from histogram_utils import plotBinHistogram
from tools import *

img = cv.imread("../imgs/lines/persp1.jpg")
img = clipImg(img, 600)

controls_window_name = 'controlls'
cv.namedWindow(controls_window_name)

tresh_param1 = 50
tresh_param2 = 150
rho_val = 1
treshold_val = 100


def set_param1(x):
    global tresh_param1
    tresh_param1 = x


def set_param2(x):
    global tresh_param2
    tresh_param2 = x


def set_rho(x):
    global rho_val
    rho_val = max(x, 1)


def set_treshold(x):
    global treshold_val
    treshold_val = x


cv.createTrackbar('param 1', controls_window_name, tresh_param1, 500, set_param1)
cv.createTrackbar('param 2', controls_window_name, tresh_param2, 500, set_param2)
cv.createTrackbar('rho', controls_window_name, rho_val, 20, set_rho)
cv.createTrackbar('tresh', controls_window_name, treshold_val, 360, set_treshold)

overlay = np.zeros(img.shape, dtype=np.uint8)
while True:

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, tresh_param1, tresh_param2, apertureSize=3)

    cv.imshow('edges', edges)

    lines = cv.HoughLinesP(edges, rho_val, np.pi / 360, treshold_val, minLineLength=10, maxLineGap=10)
    if lines is None:
        lines = []
    overlay[:, :, :] = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output = cv.add(img, overlay)
    cv.imshow('img', output)
    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
