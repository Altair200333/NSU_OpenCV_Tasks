from cv2 import cv2 as cv
import numpy as np
from tools import *
from imutils import contours
from matplotlib import pyplot as plt


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


img = cv.imread('../imgs/faces/faces.jpg')

width = img.shape[1]
height = img.shape[0]

methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF',
           'cv.TM_SQDIFF_NORMED']

method = eval(methods[0])

drawing = False
start_x = 0
start_y = 0

end_x = 0
end_y = 0

Finished = False


def onMouse(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y, Finished

    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        start_x = x
        start_y = y
        end_x = x
        end_y = y
        Finished = False

    elif event == cv.EVENT_LBUTTONUP:
        # start_x = end_x
        # start_y = end_y
        drawing = False
        Finished = True

    if event == cv.EVENT_MOUSEMOVE and drawing:
        end_x = x
        end_y = y
        # cv.rectangle(img, (start_x, start_y), (x, y), (200,200,200))
        pass


cv.namedWindow('img')
cv.setMouseCallback('img', onMouse)

gamma_val = 1
bright_val = 0
contrast_val = 0
angle = 0
scale_percent = 100

noise_range = 200


def createNoise(img):
    global noise_range
    noise = np.random.uniform(-noise_range, noise_range, img.shape[0] * img.shape[1])
    return np.int32(noise.reshape(img.shape[0], img.shape[1]))


def noiseImage(img):
    noise = createNoise(img)
    result = np.uint8(np.clip(noise + np.int32(img), 0, 255))
    return result


def gammaSet(x):
    global gamma_val
    gamma_val = x * 0.01 + 1


def brightset(x):
    global bright_val
    bright_val = x


def contrastset(x):
    global contrast_val
    contrast_val = x - 127


def angle_set(x):
    global angle
    angle = x


def scale_set(x):
    global scale_percent
    scale_percent = x


def noise_set(x):
    global noise_range
    noise_range = x


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


cv.namedWindow("controls")
cv.createTrackbar('gamma', 'controls', 0, 1000, gammaSet)
cv.createTrackbar('Brightness', 'controls', 0, 255, brightset)
cv.createTrackbar('Contrast', 'controls', 127, 255, contrastset)
cv.createTrackbar('Angle', 'controls', 0, 180, angle_set)
cv.createTrackbar('Scale', 'controls', 100, 100, scale_set)
cv.createTrackbar('Noise', 'controls', 1, 300, noise_set)

original_g = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

while True:
    canvas2 = img.copy()

    if drawing and not Finished:
        cv.rectangle(canvas2, (start_x, start_y), (end_x, end_y), (100, 200, 10), 1)

    cv.imshow('img', canvas2)

    canvas = apply_brightness_contrast(img, bright_val, contrast_val)
    canvas = rotate_image(canvas, angle)

    width = int(canvas.shape[1] * scale_percent / 100)
    height = int(canvas.shape[0] * scale_percent / 100)
    dsize = (width, height)
    canvas = cv.resize(canvas, dsize)

    gray = cv.cvtColor(canvas, cv.COLOR_RGB2GRAY)
    gray = noiseImage(gray)

    c = gray.copy()
    if Finished and abs(end_x - start_x) > 10 and abs(end_y - start_y) > 10:
        x0 = min(start_x, end_x)
        x1 = max(start_x, end_x)

        y0 = min(start_y, end_y)
        y1 = max(start_y, end_y)

        template = original_g[y0: y1, x0: x1]  # cv.imread('../imgs/faces/face.jpg', 0)
        w, h = template.shape[::-1]
        cv.imshow('template', clipImg(template, 300))

        res = cv.matchTemplate(gray, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        c = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        cv.rectangle(c, top_left, bottom_right, (255, 10, 10), 2)

    cv.imshow('canvas', c)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
