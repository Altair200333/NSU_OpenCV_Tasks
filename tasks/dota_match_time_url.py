from cv2 import cv2 as cv
import numpy as np
from match_time_from_url import *


URL = "ewin-challenge-cup-season-2/group-stage/digital-vs-crocodile-412289"

(time, region, boxes), img = match_time_url(URL)

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
