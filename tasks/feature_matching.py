import cv2 as cv
import numpy as np

from image_transform import *
from tools import *
from color_correction import *

img = cv.imread("../imgs/lines/building1.jpeg")
img = clipImg(img, 600)


def dst(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


capturedPoint = -1
radius = 30
points = [[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]]


def draw_control_points(canvas):
    for point in points:
        cv.circle(canvas, point, radius, (0, 50, 200), 2)


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


cv.namedWindow('warped')
cv.setMouseCallback('warped', onMouse)

canvas = np.zeros(img.shape, dtype=np.uint8)

orb = cv.ORB_create()
matcher = cv.BFMatcher()


def getHits(trainKeypoints, ids):
    hits = []
    for match in ids:
        hits.append(trainKeypoints[match])
    return hits


def drawHits(canvas, hits):
    for hit in hits:
        cv.circle(canvas, np.int32(hit.pt), 6, (20, 20, 200), thickness=2)


while True:
    canvas[:, :, :] = 0

    transformed = warpImage(img, points[0], points[1], points[2], points[3])

    queryKeypoints, queryDescriptors = orb.detectAndCompute(cv.cvtColor(img, cv.COLOR_BGR2GRAY), None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(cv.cvtColor(transformed, cv.COLOR_BGR2GRAY), None)

    matches = matcher.match(queryDescriptors, trainDescriptors)

    matchCanvas = cv.drawMatches(img, queryKeypoints, transformed, trainKeypoints, matches[:20], None,
                                 matchColor=(200, 20, 20), singlePointColor=(20, 200, 20))

    hits = getHits(trainKeypoints, list(match.trainIdx for match in matches))
    drawHits(canvas, hits)


    matchCanvas = clipImg(matchCanvas, 1400)

    cv.imshow('matches', matchCanvas)

    draw_control_points(canvas)
    cv.imshow('warped', cv.add(transformed, canvas))
    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
