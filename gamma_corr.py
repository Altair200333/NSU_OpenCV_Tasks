import cv2 as cv
import numpy as np
from tools import *
from histogram_utils import *
from matplotlib import pyplot as plt

img = cv.imread("imgs/lena.png")
img = clipImg(img, 600)


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(image, table)


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


gamma_val = 1
bright_val = 0
contrast_val = 0


def gammaSet(x):
    global gamma_val
    gamma_val = x * 0.01 + 1


def brightset(x):
    global bright_val
    bright_val = x


def contrastset(x):
    global contrast_val
    contrast_val = x - 127

cv.namedWindow("img")
cv.createTrackbar('gamma', 'img', 0, 1000, gammaSet)
cv.createTrackbar('Brightness', 'img', 0, 255, brightset)
cv.createTrackbar('Contrast', 'img', 127, 255, contrastset)

histogram_canvas = np.zeros((200, 256, 3), np.uint8)

img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])

# convert the YUV image back to RGB format
img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
cv.imshow("auto", img_output)
while True:
    corr = adjust_gamma(img, gamma_val)
    br_cont = apply_brightness_contrast(img, bright_val, contrast_val)

    cv.imshow("img", img)
    cv.imshow("br_cor", br_cont)
    cv.imshow("gamma_corrected", corr)

    histogram_canvas[:, :, :] = 70
    plotBinHistogram(cv.cvtColor(br_cont, cv.COLOR_BGR2GRAY), histogram_canvas)
    cv.imshow("histogram", histogram_canvas)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
