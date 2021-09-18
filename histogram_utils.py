import cv2 as cv
import numpy as np


def prepareHistData(img, dst):
    # generate histogram
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])

    # scale it to fit the shape of "canvas"
    hist = hist / hist.max() * dst.shape[0]

    # now append X axis by stacking transposed histogram and linspace
    stack = np.vstack((np.linspace(0, dst.shape[1], 256), hist.reshape(-1))).T

    # this data is still flipped; unflip it
    stack[:, 1] = dst.shape[0] - stack[:, 1]

    return stack


def plotBinHistogram(img, dst, color=(0, 255, 255)):
    stack = prepareHistData(img, dst)
    bin_w = dst.shape[1] / 255

    # bruh this python loop
    for i in range(256):
        cv.rectangle(dst, (np.int32(stack[i][0] - bin_w/2), np.int32(stack[i][1])),
                     (np.int32(stack[i][0] + bin_w/2), np.int32(dst.shape[0])), color, -1)


def plotLineHistogram(img, dst, color=(0, 255, 255)):
    stack = prepareHistData(img, dst)

    cv.polylines(dst, [np.int32(stack)], False, color)
