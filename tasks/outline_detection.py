import cv2 as cv
import numpy as np
from tools import *
from imutils import contours

paths = [
    '../imgs/im3.jpg',
    "../imgs/im4.jpg",
    "../imgs/im5.jpg",
    "../imgs/shapes.jpg",
]

img = cv.imread(paths[3])
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


cv.createTrackbar('treshold 1', controlls_window_name, treshold1, 800, set_treshold1)
cv.createTrackbar('treshold 2', controlls_window_name, treshold2, 800, set_treshold2)

canvas = np.zeros(img.shape, dtype=np.uint8)
overlay = np.zeros(img.shape, dtype=np.uint8)
approx_cntrs = np.zeros(img.shape, dtype=np.uint8)


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


def filter_contours(cntrs, tresh=100):
    filtered = []
    approxes = []
    for c in cntrs:
        hull = cv.convexHull(c)

        area = cv.contourArea(hull)
        approx = approx_contour(c)
        if area > tresh:
            filtered.append(c)
            approxes.append(approx)
    return filtered, approxes


def approx_contour(cntr):
    perimeter = cv.arcLength(cntr, True)
    approx = cv.approxPolyDP(cntr, 0.02 * perimeter, True)
    return approx


def vect_len(v):
    v = np.reshape(v, -1)
    return np.sqrt(v[0] ** 2 + v[1] ** 2)


def getPt(x):
    x = np.reshape(x, -1)
    return [x[0], x[1]]


def diff(x, y):
    x = x.ravel()
    y = y.ravel()
    return [x[0] - y[0], x[1] - y[1]]


def classify_shape(cntr, approx):

    if len(approx) == 3:
        shape = "triangle"

    elif len(approx) == 4:
        (x, y, w, h) = cv.boundingRect(approx)
        fig_area = cv.contourArea(approx)
        approx_area = vect_len(diff(approx[0], approx[1])) ** 2

        ar = fig_area / approx_area
        shape = "square" if 0.9 <= ar <= 1.2 else "rectangle"

    else:
        M = cv.moments(approx)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        radius = vect_len(approx[0] - (cX, cY))

        approx_area = np.pi * (radius ** 2)
        fig_area = cv.contourArea(approx)
        ar = approx_area / fig_area

        if 0.9 <= ar <= 1.2:
            shape = "circle"
        elif 0.5 <= ar <= 2.0:
            shape = "ellipse"
        else:
            shape = "unknown"
    # return the name of the shape
    return shape


while True:
    blurred = cv.blur(img, (3, 3))
    edges_canny = cv.Canny(blurred, treshold1, treshold2)

    contours, hierarchy = cv.findContours(edges_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    contours, approxs = filter_contours(contours)
    canvas[:, :, :] = 0
    overlay[:, :, :] = 0
    approx_cntrs[:, :, :] = 0

    draw_bounding_boxes(canvas)

    cv.drawContours(overlay, contours, -1, (50, 255, 0), 2)
    cv.drawContours(approx_cntrs, approxs, -1, (0, 20, 255), 2)

    for idx, c in enumerate(contours):
        x, y, w, h = cv.boundingRect(c)
        # point in bounding box
        result = x <= x_pos <= x + w and y <= y_pos <= y + h
        if result:
            hull = cv.convexHull(c)

            cv.drawContours(approx_cntrs, [hull], 0, (255,10,10))

            area = cv.contourArea(hull)

            fig_type = 'unknown'
            try:
                fig_type = classify_shape(c, approxs[idx])
            except:
                pass

            draw_countour_bounds(c, canvas, (244, 0, 0))
            area_ratio = area * 100 / total_area
            cv.putText(canvas, 'area: ' + "{:.1f}".format(area_ratio) + '%',
                       (np.int32(x + w / 2 - 40), np.int32(y + h / 2)), font, 0.6, fontColor, thickness)

            size_label = 'small'
            if area_ratio < 1:
                size_label = 'small'
            elif area_ratio < 5:
                size_label = 'medium'
            else:
                size_label = 'large'

            cv.putText(canvas, size_label,
                       (np.int32(x + w / 2 - 40), np.int32(y + h / 2) + 15), font, 0.6, fontColor, thickness)

            cv.putText(canvas, 'type: ' + fig_type,
                       (np.int32(x + w / 2 - 40), np.int32(y + h / 2 + 30)), font, 0.6, fontColor, thickness)

    cv.imshow('img', cv.add(img, overlay))
    cv.imshow('approx', approx_cntrs)

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
