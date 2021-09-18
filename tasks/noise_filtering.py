import cv2 as cv
import numpy as np

from histogram_utils import plotBinHistogram
from tools import *

img = cv.imread("../imgs/portraits/yana.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = clipImg(img, 600)


def create_noise(img, param, uniform: bool = True):
    noise = np.zeros(img.shape)
    if uniform:
        cv.randu(noise, -param, param)
    else:
        cv.randn(noise, 0, param)
    return noise


def create_noise_img(img, param, uniform: bool = True):
    noise = create_noise(img, param, uniform)
    result = np.clip(cv.add(np.float64(img), noise), 0, 255)

    return np.uint8(result)


noise = create_noise_img(img, 20, False)

current_method = 0
methods_count = 4

uniform_noise = 0
noise_scale = 20


def next_method(x):
    global current_method, methods_count
    current_method = x % methods_count


def get_noise_state():
    global uniform_noise
    return False if uniform_noise == 0 else True


def filter(img, method):
    filtered_img = img
    if method == 0:
        filtered_img = cv.blur(img, (3, 3))
    if method == 1:
        filtered_img = cv.medianBlur(img, 3)
    if method == 2:
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel /= (kernel_size * kernel_size)
        filtered_img = cv.filter2D(img, -1, kernel)
    if method == 3:
        filtered_img = img

    return filtered_img


controlls_window_name = 'controlls'
cv.namedWindow(controlls_window_name)


def set_noise_range(x):
    global noise, img, noise_scale
    noise_scale = x
    regenerate_noise_image()


def regenerate_noise_image():
    global noise, noise_scale, img
    noise = create_noise_img(img, noise_scale, get_noise_state())


def set_uniform(x):
    global uniform_noise, noise, img, noise_scale
    uniform_noise = x
    regenerate_noise_image()


# set noise range; std_dev for normal; [-range, range] for uniform
cv.createTrackbar('ns range', controlls_window_name, 20, 150, set_noise_range)
cv.createTrackbar('uniform', controlls_window_name, 0, 1, set_uniform)
cv.createTrackbar('method', controlls_window_name, current_method, methods_count - 1, next_method)

histogram_canvas = np.zeros((200, 400, 3), np.uint8)

while True:

    cv.imshow("noise", cv.convertScaleAbs(noise))

    filtered = filter(noise, current_method)
    cv.imshow('filtered', filtered)

    histogram_canvas[:, :, :] = 70
    plotBinHistogram(filtered, histogram_canvas)
    cv.imshow("histogram", histogram_canvas)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
