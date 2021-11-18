import cv2 as cv
import numpy as np
from tools import *
from imutils import contours

paths = [
    '../imgs/im3.jpg',  # 0 - клюшка
    "../imgs/im5.jpg",  # 1 - планета
    "../imgs/im4.jpg",  # 2 - микро клюшка
    "../imgs/triangle.jpg",  # 3 - треуг
    "../imgs/rect.jpg",  # 4 - прямоуг
]

img = cv.imread(paths[1])
img = clipImg(img, 600)

controlls_window_name = 'controlls'
cv.namedWindow(controlls_window_name)


def pass_callback(x):
    pass


total_area = img.shape[0] * img.shape[1]

width = img.shape[1]
height = img.shape[0]

# text params
font = cv.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (200, 100, 255)
thickness = 1


def approx_contour(cntr):
    perimeter = cv.arcLength(cntr, True)
    approx = cv.approxPolyDP(cntr, 0.02 * perimeter, True)
    return approx


def filter_contours(cntrs, tresh=0.002):
    global area_ratio
    filtered = []
    approximate_cntrs = []

    for c in cntrs:
        hull = cv.convexHull(c)

        area = cv.contourArea(hull)
        approx = approx_contour(c)

        if area / total_area > tresh:
            filtered.append(c)
            approximate_cntrs.append(approx)
    return filtered, approximate_cntrs


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


treshold1 = 100
treshold2 = 100

# find optimal params

grades = [0.7, 0.3]


def classify_shape(approx):
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


def classify_area(cntr):
    global total_area, grades, width, height

    x, y, w, h = cv.boundingRect(hull)

    area = cv.contourArea(cntr)

    min_dim = min(width, height)
    max_ext = max(w, h)
    ratio = max_ext / min_dim
    if max_ext / min_dim >= grades[0]:
        return ratio, 'large'
    elif max_ext / min_dim >= grades[1]:
        return ratio, 'medium'
    else:
        return ratio, 'small'

    ratio = area / total_area
    size = 'large'
    if ratio >= grades[0]:
        size = 'large'
    elif grades[1] <= ratio < grades[0]:
        size = 'medium'
    else:
        size = 'small'

    return ratio, size


def getContours(img, t1, t2):
    blurred = cv.blur(img, (3, 3))
    edges_canny = cv.Canny(blurred, t1, t2)

    contours, hierarchy = cv.findContours(edges_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours, approxs = filter_contours(contours)

    total_contour = np.vstack(contours)
    hull = cv.convexHull(total_contour)

    return hull, total_contour


max_ratio = -1
for i in range(10):
    for j in range(10):
        t1 = np.int0(i * 255 / 10)
        t2 = np.int0(j * 255 / 10)

        hull, total_contour = getContours(img, t1, t2)
        area_ratio, size_label = classify_area(hull)

        if area_ratio > max_ratio:
            max_ratio = area_ratio

            treshold1 = t1
            treshold2 = t2
mode = 0
modes_count = 2


def nextMode(x):
    global mode
    mode = (mode + x) % modes_count


def set_treshold1(x):
    global treshold1
    treshold1 = x


def set_treshold2(x):
    global treshold2
    treshold2 = x


cv.createTrackbar('treshold 1', controlls_window_name, treshold1, 800, set_treshold1)
cv.createTrackbar('treshold 2', controlls_window_name, treshold2, 800, set_treshold2)

canvas = np.zeros(img.shape, dtype=np.uint8)
overlay = np.zeros(img.shape, dtype=np.uint8)
approx_cntrs = np.zeros(img.shape, dtype=np.uint8)


def flatten(cntrs):
    list_of_pts = []
    for ctr in cntrs:
        list_of_pts += [pt[0] for pt in ctr]
    return np.asarray(list_of_pts)


def approx_contour(cntr):
    perimeter = cv.arcLength(cntr, True)
    approx = cv.approxPolyDP(cntr, 0.02 * perimeter, True)
    return approx


while True:
    canvas[:, :, :] = 0
    overlay[:, :, :] = 0
    approx_cntrs[:, :, :] = 0

    hull, total_contour = getContours(img, treshold1, treshold2)

    area_ratio, size_label = classify_area(hull)
    type = classify_shape(approx_contour(hull))

    if mode == 1:
        cv.drawContours(overlay, total_contour, -1, (50, 255, 0), 1)
    else:
        x, y, w, h = cv.boundingRect(hull)
        cv.putText(overlay, 'area: ' + "{:.1f}".format(area_ratio * 100) + '%',
                   (np.int32(x + w / 2 - 40), np.int32(y + h / 2)), font, 0.6, fontColor, thickness)
        cv.putText(overlay, size_label,
                   (np.int32(x + w / 2 - 40), np.int32(y + h / 2) + 15), font, 0.6, fontColor, thickness)
        cv.putText(overlay, type,
                   (np.int32(x + w / 2 - 40), np.int32(y + h / 2) + 32), font, 0.6, fontColor, thickness)

    cv.drawContours(overlay, [hull], 0, (255, 10, 10))

    cv.imshow('img', img)
    cv.imshow('overlay', cv.addWeighted(img, 0.3, overlay, 0.7, 0.0))

    k = cv.waitKey(1) & 0xFF

    if k == ord('q'):
        nextMode(-1)
    if k == ord('e'):
        nextMode(1)
    if k == 27:
        break
