import cv2 as cv
import numpy as np

from image_transform import *
from tasks.tracking_tools import *
from tools import *
from color_correction import *


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


img1 = cv.imread("../imgs/portraits/hana.jpg")
img1 = clipImg(img1, 600)

img2 = cv.imread("../imgs/portraits/hana.jpg")
img2 = clipImg(img2, 600)
img2 = create_noise_img(img2, 50, True)


def dst(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


capturedPoint = -1
radius = 30
points = [[0, 0], [img2.shape[1], 0], [img2.shape[1], img2.shape[0]], [0, img2.shape[0]]]


def draw_control_points(canvas):
    for point in points:
        cv.circle(canvas, point, radius, (0, 50, 200), 2)


def onMouse(event, x, y, flags, param):
    global points, capturedPoint

    x = np.clip(x, 0, img2.shape[1] - 1)
    y = np.clip(y, 0, img2.shape[0] - 1)

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

canvas = np.zeros(img2.shape, dtype=np.uint8)

orb = cv.ORB_create()
matcher = cv.BFMatcher()


def drawHits(canvas, hits):
    for hit in hits:
        cv.circle(canvas, np.int32(hit.pt), 6, (20, 20, 200), thickness=2)

canvas[:, :, :] = 0

transformed = warpImage(img2, points[0], points[1], points[2], points[3])

queryKeypoints, queryDescriptors = orb.detectAndCompute(cv.cvtColor(img1, cv.COLOR_BGR2GRAY), None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(cv.cvtColor(transformed, cv.COLOR_BGR2GRAY), None)

matches = matcher.match(queryDescriptors, trainDescriptors)

fitted = [i for i in range(len(matches)) if (matches[i].distance < 2)]# np.where(np.asarray(matches).distance < 2)[0]

percentage = len(fitted)/len(matches)
print("matched " + str(percentage))
matchCanvas = cv.drawMatches(img1, queryKeypoints, transformed, trainKeypoints, np.asarray(matches)[fitted], None, matchColor=(200, 20, 20), singlePointColor=(20, 200, 20))

hits = getHits(trainKeypoints, list(match.trainIdx for match in matches))
drawHits(canvas, hits)

matchCanvas = clipImg(matchCanvas, 1000)

while True:


    cv.imshow('matches', matchCanvas)

    #draw_control_points(canvas)
    #cv.imshow('warped', cv.add(transformed, canvas))
    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
