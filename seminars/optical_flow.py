from cv2 import cv2 as cv
import numpy as np
from tools import *

paths = [
    '../videos/traffic3.mp4',
    '../videos/traffic4.mp4',
    '../videos/traffic2_1.mp4'
]
frames, totalFrames = read_frames(paths[1])

print(totalFrames)

print("calculating flow")

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

old_gray = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

good_new_list = []
good_old_list = []

for i in range(1, totalFrames):
    frame_gray = cv.cvtColor(frames[i], cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    good_new_list.append(good_new)
    good_old_list.append(good_old)

    p0 = good_new.reshape(-1, 1, 2)
    old_gray = frame_gray.copy()




print("flow calculated")

mask = np.zeros_like(frames[0])

frame_number = 0


def set_frame_number(x):
    global frame_number, totalFrames
    frame_number = x % totalFrames


cv.namedWindow('control')
cv.createTrackbar('frame', 'control', 1, int(totalFrames) - 1, set_frame_number)

color = np.random.randint(0, 255, (100, 3))


def find_points_between(frame1, frame2):
    global feature_params

    old_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    frame_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    return good_new, good_old

playing = False

while True:
    if playing:
        frame_number = (frame_number + 1) % totalFrames

    frame = frames[frame_number].copy()

    for j in range(max(0, frame_number - 40), frame_number):
        good_new = good_new_list[j]
        good_old = good_old_list[j]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv.line(frame, np.int0((a, b)), np.int0((c, d)), (0, 20, 200), 1)
            frame = cv.circle(frame, np.int0((a, b)), 1, (0, 20, 200), -1)

    #img = cv.add(frame, mask)

    cv.imshow('img', frame)

    k = cv.waitKey(1) & 0xFF

    if k == ord('w'):
        playing = not playing
    if k == 27:
        break
