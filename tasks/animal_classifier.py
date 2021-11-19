from cv2 import cv2 as cv
import numpy as np
from tools import *
from imutils import contours
from matplotlib import pyplot as plt
from functools import cmp_to_key
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import os, os.path

# img = cv.imread('../imgs/an/badger/b1.jpg', 0)
# img2 = cv.imread('../imgs/an/chipmunk/c1.jpg', 0)

X = []
y = []

size = 200
for f in os.listdir('../imgs/an/badger'):
    img = cv.imread('../imgs/an/badger/' + f, 0)
    img = cv.resize(img, (size, size))
    data = img.ravel()
    X.append(data)
    y.append(0)
    # print(data)

for f in os.listdir('../imgs/an/badger'):
    img = cv.imread('../imgs/an/badger/' + f, 0)
    img = cv.resize(img, (size, size))
    data = img.ravel()
    X.append(data)
    y.append(1)
    # print(data)

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X, y)

pred = clf.predict([X[0]])

print(pred)
