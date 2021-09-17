import cv2 as cv
import numpy as np
from tools import *

img = cv.imread("imgs/lena.png")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = clipImg(img, 600)

cv.namedWindow("img")

noise_range = 200


def createNoise(img):
    global noise_range
    noise = np.random.uniform(-noise_range, noise_range, img.shape[0] * img.shape[1])
    return np.int32(noise.reshape(img.shape[0], img.shape[1]))

def noiseImage(img):
    noise = createNoise(img)
    result = np.uint8(np.clip(noise + np.int32(img), 0, 255))
    return result


noise_imgs_count = 15
noise_images = []
for i in range(noise_imgs_count):
    noise_images.append(noiseImage(img))

shownId = 0


def setShownId(x):
    global shownId
    shownId = x


cv.createTrackbar('show img[i]', 'img', 0, noise_imgs_count - 1, setShownId)

while True:
    cv.imshow("img", noise_images[shownId])

    filter = np.zeros(img.shape, dtype=np.float64)

    for i in range(noise_imgs_count):
        filter += np.float64(noise_images[i])
    filter[:, :] = np.float64(filter[:, :])/noise_imgs_count
    cv.imshow("filtered", np.uint8(filter))

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
