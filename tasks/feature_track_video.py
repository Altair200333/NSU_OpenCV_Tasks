import cv2 as cv
import numpy as np

from image_transform import *
from tools import *
from color_correction import *

frame_size = 600

cap = cv.VideoCapture('../videos/building.mp4')
totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)


if cap.isOpened() == False:
    print("Error opening video  file")

frames = []
for i in range(min(100, int(totalFrames))):
    ret, frame = cap.read()
    if not ret:
        break
    frame = clipImg(frame, frame_size)
    frames.append(frame)

totalFrames = len(frames)
frame_number = 0


def set_frame_number(x):
    global frame_number, totalFrames, cap
    frame_number = x % totalFrames

    # cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)


orb = cv.ORB_create()
matcher = cv.BFMatcher()

set_frame_number(0)

first_frame = frames[0]
first_frame = clipImg(first_frame, frame_size)

queryKeypoints, queryDescriptors = orb.detectAndCompute(cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY), None)

match_window_name = 'matching'
cv.namedWindow(match_window_name)
cv.namedWindow('controls')

cv.createTrackbar('frame', 'controls', 0, int(totalFrames) - 1, set_frame_number)

while True:

    frame = frames[int(frame_number)]

    trainKeypoints, trainDescriptors = orb.detectAndCompute(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), None)
    matches = matcher.match(queryDescriptors, trainDescriptors)

    matchCanvas = cv.drawMatches(first_frame, queryKeypoints, frame, trainKeypoints, matches[:20], None,
                                 matchColor=(200, 20, 20), singlePointColor=(20, 200, 20))
    matchCanvas = clipImg(matchCanvas, 1400)

    cv.imshow(match_window_name, matchCanvas)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
