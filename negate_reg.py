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

    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

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


def plotHist(dst, hist):
    stack = np.vstack((np.linspace(0, 256, 256), hist.reshape(-1))).T
    stack[:, 1] = dst.shape[0] - stack[:, 1]

    cv.polylines(dst, [np.int32(stack)], True, (0, 255, 255))


histogram_canvas = np.zeros((200, 256, 3), np.uint8)
cv.setMouseCallback('win', onMouse)


def histogram_result():
    histogram_canvas[:, :, :] = 0
    hist = cv.calcHist([result[:, :, 1]], [0], None, [256], [0, 256])
    plotHist(histogram_canvas, hist / hist.max() * 100)
    cv.imshow("hist", histogram_canvas)

gradient_map = np.ones((256, 256, 3), np.uint8)
for i in range(0, 255):
    for j in range(0, 255):
        gradient_map[i, j] = [200, j, 255-i]

cv.imshow("grad", cv.cvtColor(gradient_map, cv.COLOR_Lab2BGR) )
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

    output = np.ones((256, 256, 3), np.uint8)*100
    lab_result = cv.cvtColor(result, cv.COLOR_BGR2LAB)

    output[lab_result[:, :, 1], 255 - lab_result[:, :, 2], :] = gradient_map[lab_result[:, :, 1], 255 - lab_result[:, :, 2], :]
    output = cv.cvtColor(output, cv.COLOR_Lab2BGR)
    cv.imshow('lab_spread', output)

    histogram_result()

    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):
        transform_mode = (transform_mode - 1) % 3
    if k == ord('e'):
        transform_mode = (transform_mode + 1) % 3
    if k == 27:
        break
