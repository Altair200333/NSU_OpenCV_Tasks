from cv2 import cv2 as cv
import numpy as np
from tools import *
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2

with torch.no_grad():
    model = model.cuda()
    #model(frames[0].cuda())


exit()
frames, totalFrames = read_frames('../videos/traffic2_1.mp4')

frame_number = 0


def set_frame_number(x):
    global frame_number, totalFrames
    frame_number = x % totalFrames


cv.namedWindow('control')
cv.createTrackbar('frame', 'control', 1, int(totalFrames) - 1, set_frame_number)

#448
while True:

    cv.imshow('img', frames[frame_number])

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break
