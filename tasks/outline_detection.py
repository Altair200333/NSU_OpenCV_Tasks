import cv2 as cv
import numpy as np
from tools import *
from imutils import contours

img = cv.imread('../imgs/shapes.jpg')
img = clipImg(img, 600)

controlls_window_name = 'controlls'
edges_window_name = 'edges'
cv.namedWindow(controlls_window_name)
cv.namedWindow(edges_window_name)


def pass_callback(x):
    pass


treshold1 = 100
treshold2 = 100


def set_treshold1(x):
    global treshold1
    treshold1 = x


def set_treshold2(x):
    global treshold2
    treshold2 = x


mode = 0
modes_count = 2


def nextMode(x):
    global mode
    mode = (mode + x) % modes_count


cv.createTrackbar('treshold 1', controlls_window_name, 200, 500, set_treshold1)
cv.createTrackbar('treshold 2', controlls_window_name, 220, 500, set_treshold2)

canvas = np.zeros(img.shape, dtype=np.uint8)
overlay = np.zeros(img.shape, dtype=np.uint8)


def draw_bounding_boxes(canvas):
    for c in contours:
        draw_countour_bounds(c, canvas)


def draw_countour_bounds(c, canvas, color=(0, 255, 0)):
    x, y, w, h = cv.boundingRect(c)
    cv.rectangle(canvas, (x, y), (x + w, y + h), color, 1)


x_pos = 1
y_pos = 1


def onMouse(event, x, y, flags, param):
    global x_pos, y_pos
    x = np.clip(x, 0, img.shape[1] - 1)
    y = np.clip(y, 0, img.shape[0] - 1)

    x_pos = x
    y_pos = y


cv.setMouseCallback(edges_window_name, onMouse)

total_area = img.shape[0] * img.shape[1]

font = cv.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (200, 100, 255)
thickness = 1

while True:
    edges_canny = cv.Canny(img, treshold1, treshold2)

    contours, hierarchy = cv.findContours(edges_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    canvas[:, :, :] = 0
    overlay[:, :, :] = 0

    draw_bounding_boxes(canvas)

    cv.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        # point in bounding box
        result = x <= x_pos <= x + w and y <= y_pos <= y + h
        if result:
            area = cv.contourArea(c)
            cv.putText(canvas, 'area: '+"{:.1f}".format(area*100/total_area)+'%', (np.int32(x + w / 2 - 40), np.int32(y + w / 2 - 30)), font, 0.6, fontColor,thickness)
            draw_countour_bounds(c, canvas, (244, 0, 0))

    cv.imshow('img', cv.add(img, overlay))

    result = canvas
    if mode == 0:
        result = cv.add(cv.cvtColor(edges_canny, cv.COLOR_GRAY2BGR), canvas)
    else:
        result = cv.cvtColor(edges_canny, cv.COLOR_GRAY2BGR)

    cv.imshow(edges_window_name, result)

    k = cv.waitKey(1) & 0xFF

    if k == ord('q'):
        nextMode(-1)
    if k == ord('e'):
        nextMode(1)
    if k == 27:
        break
