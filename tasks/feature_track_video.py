import cv2 as cv
import numpy as np

from image_transform import *
from tools import *
from color_correction import *
from tasks.tracking_tools import *

frame_size = 600

cap = cv.VideoCapture('../videos/building2_hd.mp4')
totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)

if cap.isOpened() == False:
    print("Error opening video  file")

frames = []
for i in range(min(30, int(totalFrames))):
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

canvas = np.zeros((500, 800, 3), dtype=np.uint8)

# frame; point; position
history = np.zeros((totalFrames, len(queryKeypoints), 2))
history[:, :, :] = -1
for i in range(history.shape[1]):
    history[0, i] = queryKeypoints[i].pt

for i in range(totalFrames):
    frame = frames[int(i)]
    trainKeypoints, trainDescriptors = orb.detectAndCompute(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), None)
    matches = matcher.match(queryDescriptors, trainDescriptors)

    train_hits = getHits(trainKeypoints, list(match.trainIdx for match in matches))
    query_hits = getHits(queryKeypoints, list(match.queryIdx for match in matches))

    if i > 0:
        for j in range(history.shape[1]):
            history[i, j] = history[i - 1, j]
    # x = canvas.shape[1] / totalFrames * i
    for idx, match in enumerate(matches):
        history[i, match.queryIdx] = train_hits[idx].pt
        # cv.line(canvas, (np.int0(x), np.int0(0)), (np.int0(x), np.int0(abs(train_hits[idx].pt[0] - query_hits[idx].pt[0]))), (10, 20, 200))
        # cv.line(canvas, (np.int0(x), np.int0(0)), (np.int0(x), np.int0(abs(train_hits[idx].pt[1] - query_hits[idx].pt[1]))), (200, 20, 20))

differences = np.zeros(history.shape)
for j in range(history.shape[0]):
    if j > 0:
        differences[j] = history[j] - history[j - 1]
# print(differences)
for i in range(history.shape[0]):
    prev = differences[max(0, i - 1)]
    curr = differences[i]
    scale_x = canvas.shape[1] / history.shape[0]
    for j in range(history.shape[1]):
        cv.line(canvas, (np.int0((i - 1) * scale_x), np.int0(abs(prev[j][0]))),(np.int0(i * scale_x), np.int0(abs(curr[j][0]))), (10, 20, 200))
        cv.line(canvas, (np.int0((i - 1) * scale_x), np.int0(abs(prev[j][1]))),(np.int0(i * scale_x), np.int0(abs(curr[j][1]))), (20, 200, 20))
        # print(offset)

while True:
    # canvas[:, :, :] = 0

    frame = frames[int(frame_number)]

    trainKeypoints, trainDescriptors = orb.detectAndCompute(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), None)
    matches = matcher.match(queryDescriptors, trainDescriptors)

    matchCanvas = cv.drawMatches(first_frame, queryKeypoints, frame, trainKeypoints, matches[:20], None,
                                 matchColor=(200, 20, 20), singlePointColor=(20, 200, 20))
    matchCanvas = clipImg(matchCanvas, 1400)

    cv.imshow(match_window_name, matchCanvas)

    train_hits = getHits(trainKeypoints, list(match.trainIdx for match in matches))
    query_hits = getHits(queryKeypoints, list(match.queryIdx for match in matches))

    # center = (np.int0(canvas.shape[1]/2), np.int0(canvas.shape[0]/2))
    # for i, match in enumerate(matches):
    #    offset = np.int32(train_hits[i].pt) - np.int32(query_hits[i].pt)
    #    cv.line(canvas, center, center + offset, (200,200,200))

    cv.imshow('deviation', canvas)
    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
