from cv2 import cv2 as cv
import numpy as np

from dota_match_time import *

img = cv.imread('../imgs/dota/match3.jpg')

time, region, boxes = math_time(img)

canvas = cv.cvtColor(region, cv.COLOR_GRAY2BGR)

for box in boxes:
    cv.rectangle(canvas, box.box[0], box.box[1], (200, 20, 20))

canvas = clipImg(canvas, 500)

print(time)

while True:

    cv.imshow('time', canvas)
    cv.imshow('img', img)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
