import cv2 as cv
import numpy as np


def warpImage(img, p1, p2, p3, p4):
    input_uv = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])
    output_uv = np.float32([p1, p2, p3, p4])

    matrix = cv.getPerspectiveTransform(input_uv, output_uv)
    imgOutput = cv.warpPerspective(img, matrix, (img.shape[1], img.shape[0]), cv.INTER_CUBIC,
                                   borderMode=cv.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))
    return imgOutput

