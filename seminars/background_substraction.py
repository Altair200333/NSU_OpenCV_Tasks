from cv2 import cv2 as cv
import numpy as np
from tools import *

paths = [
'../videos/traffic1.mp4',
'../videos/traffic4.mp4'
]
frames, totalFrames = read_frames(paths[1])

print(totalFrames)

backSub = cv.createBackgroundSubtractorKNN()  # cv.createBackgroundSubtractorMOG2()

fgMask = backSub.apply(frames[0])

frame_number = 0

mode = 0
modes_count = 3


def nextMode(x):
    global mode
    mode = (mode + x) % modes_count


def set_frame_number(x):
    global frame_number, totalFrames, fgMask
    frame_number = x % totalFrames
    fgMask = backSub.apply(frames[frame_number])


def set_substractor(x):
    global backSub, fgMask
    if x == 0:
        backSub = cv.createBackgroundSubtractorKNN()
    else:
        backSub = cv.createBackgroundSubtractorMOG2()

    for i in range(10):
        fgMask = backSub.apply(frames[i])


cv.namedWindow('control')
cv.createTrackbar('frame', 'control', 0, int(totalFrames) - 1, set_frame_number)
cv.createTrackbar('substractor', 'control', 0, 1, set_substractor)

playing = False

while True:

    if playing:
        frame_number = (frame_number + 1) % totalFrames
        fgMask = backSub.apply(frames[frame_number])

    cv.imshow('FG Mask', fgMask)

    if mode == 0:
        cv.imshow('img', frames[frame_number])
    elif mode == 1:
        cv.imshow('img', cv.bitwise_and(frames[frame_number], frames[frame_number], mask=fgMask))
    elif mode == 2:
        cv.imshow('img', backSub.getBackgroundImage())

    k = cv.waitKey(10) & 0xFF

    if k == ord('w'):
        playing = not playing

    if k == ord('q'):
        nextMode(-1)
    if k == ord('e'):
        nextMode(1)

    if k == 27:
        break
