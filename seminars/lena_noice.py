import cv2 as cv
import numpy as np
from tools import *

img = cv.imread("../imgs/portraits/lena.png")
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

filtered = img

noise_imgs_count = 15
noise_images = []

shownId = 0

def setShownId(x):
    global shownId
    shownId = x

def generateNoiseImgs(x):
    global noise_imgs_count, noise_images, img, shownId
    x = max(1, x)
    noise_imgs_count = x
    noise_images.clear()
    for i in range(noise_imgs_count):
        noise_images.append(noiseImage(img))

    shownId = min(noise_imgs_count - 1, shownId)

    filter_noised()

def filter_noised():
    global filtered
    filtered = np.zeros(img.shape, dtype=np.float64)
    for i in range(noise_imgs_count):
        filtered += np.float64(noise_images[i])
    filtered[:, :] = np.float64(filtered[:, :]) / noise_imgs_count

def updateRange(x):
    global noise_range
    noise_range = x
    generateNoiseImgs(noise_imgs_count)

generateNoiseImgs(10)

cv.createTrackbar('show img[i]', 'img', 0, noise_imgs_count - 1, setShownId)
cv.createTrackbar('noise_images', 'img', 2, 100, generateNoiseImgs)
cv.createTrackbar('noise range', 'img', 20, 255, updateRange)


while True:
    cv.imshow("img", noise_images[shownId%noise_imgs_count])

    cv.imshow("filtered", np.uint8(filtered))

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
