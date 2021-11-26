from cv2 import cv2 as cv
import numpy as np
from tools import *

frames, totalFrames = read_frames('../videos/traffic3.mp4')

print(totalFrames)

backSub = cv.createBackgroundSubtractorKNN()  # cv.createBackgroundSubtractorMOG2()

fgMask = backSub.apply(frames[0])

frame_number = 0

mode = 0
modes_count = 2


def nextMode(x):
    global mode
    mode = (mode + x) % modes_count


def set_frame_number(x):
    global frame_number, totalFrames, fgMask
    frame_number = x % totalFrames
    fgMask = backSub.apply(frames[frame_number])


def set_substractor(x):
    global backSub
    if x == 0:
        backSub = cv.createBackgroundSubtractorKNN()
    else:
        backSub = cv.createBackgroundSubtractorMOG2()


cv.namedWindow('controls')
cv.createTrackbar('frame', 'controls', 0, int(totalFrames) - 1, set_frame_number)
cv.createTrackbar('substractor', 'controls', 0, 1, set_substractor)

while True:

    cv.imshow('FG Mask', fgMask)

    if mode == 0:
        cv.imshow('img', frames[frame_number])
    elif mode == 1:
        cv.imshow('img', cv.bitwise_and(frames[frame_number], frames[frame_number], mask=fgMask))

    k = cv.waitKey(1) & 0xFF

    if k == ord('q'):
        nextMode(-1)
    if k == ord('e'):
        nextMode(1)

    if k == 27:
        break
