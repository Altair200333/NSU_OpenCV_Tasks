import cv2 as cv
import numpy as np

from tools import *
from color_correction import *

img = cv.imread("../imgs/lines/check.jpg")
img = clipImg(img, 600)


def warpImage(img, p1, p2, p3, p4):
    input_uv = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])
    output_uv = np.float32([p1, p2, p3, p4])

    matrix = cv.getPerspectiveTransform(input_uv, output_uv)
    imgOutput = cv.warpPerspective(img, matrix, (img.shape[1], img.shape[0]), cv.INTER_CUBIC,
                                   borderMode=cv.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))
    return imgOutput


capturedPoint = -1
radius = 30
points = [[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]]

tomachi_canvas = np.zeros(img.shape, dtype=np.uint8)
harris_canvas = np.zeros(img.shape, dtype=np.uint8)


def resetPoint():
    global points
    points = [[0, 0], [img.shape[1] - 1, 0], [img.shape[1] - 1, img.shape[0] - 1], [0, img.shape[0] - 1]]


def dst(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def onMouse(event, x, y, flags, param):
    global points, capturedPoint

    x = np.clip(x, 0, img.shape[1] - 1)
    y = np.clip(y, 0, img.shape[0] - 1)

    if event == cv.EVENT_LBUTTONDOWN:
        for idx, point in enumerate(points):
            if dst([x, y], point) <= radius:
                capturedPoint = idx

    if event == cv.EVENT_LBUTTONUP:
        capturedPoint = -1

    if event == cv.EVENT_MOUSEMOVE and capturedPoint != -1:
        points[capturedPoint] = [x, y]


shi_tomachi_window = 'warped'
harris_window = 'har'
cv.namedWindow(shi_tomachi_window)
cv.namedWindow(harris_window)
cv.setMouseCallback(shi_tomachi_window, onMouse)
cv.setMouseCallback(harris_window, onMouse)

# warp_canvas = np.zeros(img.shape, dtype=np.uint8)

shi_tomachi_name = 'shi-tomachi'
harris_name = 'harris'

cv.namedWindow(shi_tomachi_name)
cv.namedWindow(harris_name)

corners_count = 100
corners_quality = 0.02
min_dst = 10


def set_corners_count(x):
    global corners_count
    corners_count = max(1, x)


def set_corners_quality(x):
    global corners_quality
    corners_quality = max(x / 100, 0.001)


def set_min_dst(x):
    global min_dst
    min_dst = max(1, x)


block_size = 2
k_size = 2
k_param = 2


def set_block_size(x):
    global block_size
    block_size = max(1, x)


def set_ksize(x):
    global k_size
    if x % 2 == 0:
        x += 1
    k_size = max(1, x)


def set_k(x):
    global k_param
    k_param = max(x / 100, 0.001)


# 0 - goodFeaturesToTrack; 1 - cornerHarris

def create_controls(name, mode):
    if mode == 0:
        cv.createTrackbar('corners', name, 100, 255, set_corners_count)
        cv.createTrackbar('quality', name, 2, 100, set_corners_quality)
        cv.createTrackbar('min dst', name, 10, 100, set_min_dst)
    else:
        cv.createTrackbar('block', name, 2, 50, set_block_size)
        cv.createTrackbar('ksize', name, 3, 30, set_ksize)
        cv.createTrackbar('k', name, 4, 100, set_k)


create_controls(shi_tomachi_name, 0)
create_controls(harris_name, 1)

brightness = 0
contrast = 0

def set_brightness(x):
    global brightness
    brightness = x


def set_contrast(x):
    global contrast
    contrast = x - 127

cv.namedWindow('adjustment')
cv.createTrackbar('brightness', 'adjustment', 0, 255, set_brightness)
cv.createTrackbar('contrast', 'adjustment', 127, 255, set_contrast)


def draw_control_points(canvas):
    for point in points:
        cv.circle(canvas, point, radius, (0, 50, 200), 2)


def draw_markers(canvas, corners):
    corners_int = np.int0(corners)
    for corner in corners_int:
        x, y = corner.ravel()
        cv.drawMarker(canvas, (x, y), (255, 100, 0), cv.MARKER_CROSS, 22, 2)


# 0 - points on top of image; 1 - only points
display_mode = 0


def next_display_mode(x):
    global display_mode
    display_mode = (display_mode + x) % 2


controls_overlay = np.zeros(img.shape, dtype=np.uint8)


def compose_output(img, track_data, controlls):
    global display_mode
    output = img
    if display_mode == 0:
        output = cv.add(img, track_data)
        output = cv.add(output, controlls)
    else:
        output = cv.add(track_data, controlls)

    return output


while True:

    tomachi_canvas[:, :, :] = 0
    harris_canvas[:, :, :] = 0
    controls_overlay[:, :, :] = 0

    draw_control_points(controls_overlay)

    adjusted = apply_brightness_contrast(img, brightness, contrast)

    warped_img = warpImage(adjusted, points[0], points[1], points[2], points[3])

    # ---
    gray_warped = cv.cvtColor(warped_img, cv.COLOR_BGR2GRAY)
    tomachi_corners = cv.goodFeaturesToTrack(cv.cvtColor(warped_img, cv.COLOR_BGR2GRAY), corners_count, corners_quality,
                                             min_dst,
                                             mask=cv.inRange(gray_warped, 1, 255))

    draw_markers(tomachi_canvas, tomachi_corners)

    cv.imshow(shi_tomachi_window, compose_output(warped_img, tomachi_canvas, controls_overlay))
    # ---

    harris_corners = cv.cornerHarris(gray_warped, block_size, k_size, k_param)
    harris_canvas[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

    cv.imshow(harris_window, compose_output(warped_img, harris_canvas, controls_overlay))

    # --
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(gray, corners_count, corners_quality, min_dst)
    draw_markers(tomachi_canvas, corners)

    # cv.imshow('img', cv.add(img, tomachi_canvas))

    k = cv.waitKey(1) & 0xFF

    if k == ord('q'):
        next_display_mode(-1)
    if k == ord('e'):
        next_display_mode(1)
    if k == 27:
        break
