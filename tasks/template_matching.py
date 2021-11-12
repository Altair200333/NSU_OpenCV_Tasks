from cv2 import cv2 as cv
import numpy as np
from tools import *
from imutils import contours
from matplotlib import pyplot as plt
from functools import cmp_to_key

img = cv.imread('../imgs/dota/match1.jpg', 0)
# img = clipImg(img, 600)

img2 = img.copy()
template = cv.imread('../imgs/dota/template.jpg', 0)

method = eval('cv.TM_CCOEFF_NORMED')


def find_region(img, template):
    global method
    w, h = template.shape[::-1]

    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return top_left, bottom_right, res


def find_boxes(img, template, threshold=0.9):
    global method
    w, h = template.shape[::-1]

    res = cv.matchTemplate(img, template, method)

    detected_boxes = []

    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(region_canvas, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
        detected_boxes.append((pt, (pt[0] + w, pt[1] + h)))

    return detected_boxes


top_left, bottom_right, _ = find_region(img, template)

time_region = img[top_left[1]:bottom_right[1], bottom_right[0]:bottom_right[0] + 40]

region_canvas = time_region.copy()
region_canvas = cv.cvtColor(region_canvas, cv.COLOR_GRAY2BGR)

numbers = []
for i in range(10):
    numbers.append(cv.imread('../imgs/dota/numbers/' + str(i) + '.jpg', 0))

colon_template = cv.imread('../imgs/dota/numbers/c.jpg', 0)


class Number:
    def __init__(self, num, box, isColon=False):
        self.num = num
        self.box = box
        self.isColon = isColon


registered_numbers = []

for idx, number in enumerate(numbers):
    boxes = find_boxes(time_region, number)
    for box in boxes:
        registered_numbers.append(Number(idx, box))
        cv.rectangle(region_canvas, box[0], box[1], (255, 10, 10), 0)

boxes = find_boxes(time_region, colon_template)
for box in boxes:
    registered_numbers.append(Number(-1, box, True))
    cv.rectangle(region_canvas, box[0], box[1], (255, 10, 200), 0)


def box_center(box):
    return (box[0][0] + box[1][0]) * 0.5


def compare(item1, item2):
    if box_center(item1.box) < box_center(item2.box):
        return -1
    elif box_center(item1.box) > box_center(item2.box):
        return 1
    else:
        return 0


registered_numbers = sorted(registered_numbers, key=cmp_to_key(compare))

time = ''
for num in registered_numbers:
    time += ':' if num.isColon else str(num.num)

print(time)

time_region = clipImg(region_canvas, 500)

while True:
    canvas = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    cv.rectangle(canvas, top_left, bottom_right, (255, 10, 10), 1)

    cv.imshow('reg', time_region)

    cv.imshow('img', canvas)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
