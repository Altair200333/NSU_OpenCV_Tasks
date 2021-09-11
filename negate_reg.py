import cv2 as cv
import numpy as np
from tools import *

img = cv.imread('imgs/niki.jpg')
img = clipImg(img, 600)

width = img.shape[1]
height = img.shape[0]

cv.namedWindow('win')

drawing = False
start_x = 0
start_y = 0

end_x = 0
end_y = 0

transform_mode = 0


def onMouse(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        start_x = x
        start_y = y
        end_x = x
        end_y = y
    elif event == cv.EVENT_LBUTTONUP:
        start_x = end_x
        start_y = end_y
        drawing = False

    if event == cv.EVENT_MOUSEMOVE and drawing:
        end_x = x
        end_y = y
        # cv.rectangle(img, (start_x, start_y), (x, y), (200,200,200))
        pass


cv.setMouseCallback('win', onMouse)

while True:

    overlay = np.zeros((height, width, 3), np.uint8)
    mask = np.zeros((height, width, 3), np.uint8)
    mask_inv = np.ones((height, width, 3), np.uint8) * 255
    if drawing:
        cv.rectangle(overlay, (start_x, start_y), (end_x, end_y), (200, 200, 200))

        x0 = min(start_x, end_x)
        x1 = max(start_x, end_x)

        y0 = min(start_y, end_y)
        y1 = max(start_y, end_y)

        if transform_mode == 2:
            overlay[y0 + 1:y1, x0 + 1: x1, :] = 255 - img[y0 + 1:y1, x0 + 1: x1, :]
        else:
            lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
            if transform_mode == 0:
                lab_img[:, :, 1] = lab_img[:, :, 2]
            elif transform_mode == 1:
                lab_img[:, :, 2] = lab_img[:, :, 1]
            collapsed_image = cv.cvtColor(lab_img, cv.COLOR_Lab2BGR)

            overlay[y0 + 1:y1, x0 + 1: x1, :] = collapsed_image[y0 + 1:y1, x0 + 1: x1, :]

        mask[y0:y1 + 1, x0: x1 + 1, :] = 255
        mask_inv = cv.bitwise_not(mask)

    result = cv.bitwise_and(img, mask_inv)
    inv = cv.bitwise_and(overlay, mask)
    result = cv.add(result, inv)

    cv.imshow('win', result)

    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):
        transform_mode = (transform_mode - 1) % 3
    if k == ord('e'):
        transform_mode = (transform_mode + 1) % 3
    if k == 27:
        break
