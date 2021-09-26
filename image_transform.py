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


def warpImageFit(img, s1, s2, s3, s4, dst):
    input_uv = np.float32([s1, s2, s3, s4])
    output_uv = np.float32([[0, 0], [dst.shape[1], 0], [dst.shape[1], dst.shape[0]], [0, dst.shape[0]]])

    matrix = cv.getPerspectiveTransform(input_uv, output_uv)
    imgOutput = cv.warpPerspective(img, matrix, (dst.shape[1], dst.shape[0]), cv.INTER_CUBIC,
                                   borderMode=cv.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))
    return imgOutput


def transformPt(p, matrix):
    px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / ((matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / ((matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    p_after = (int(px), int(py))  # after transformation