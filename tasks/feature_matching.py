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
img2 = create_noise_img(img2, 40, False)


def dst(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

orb = cv.ORB_create()
matcher = cv.BFMatcher()


queryKeypoints, queryDescriptors = orb.detectAndCompute(cv.cvtColor(img1, cv.COLOR_BGR2GRAY), None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(cv.cvtColor(img2, cv.COLOR_BGR2GRAY), None)

matches = matcher.match(queryDescriptors, trainDescriptors)

fitted = [i for i in range(len(matches)) if (matches[i].distance < 2)]# np.where(np.asarray(matches).distance < 2)[0]

percentage = len(fitted)/len(matches)
print("matched " + str(percentage))
matchCanvas = cv.drawMatches(img1, queryKeypoints, img2, trainKeypoints, np.asarray(matches)[fitted], None, matchColor=(200, 20, 20), singlePointColor=(20, 200, 20))

hits = getHits(trainKeypoints, list(match.trainIdx for match in matches))

matchCanvas = clipImg(matchCanvas, 1000)

while True:


    cv.imshow('matches', matchCanvas)


    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
