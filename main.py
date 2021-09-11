# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import matplotlib.pyplot as plt


def nothing(x):
    pass

print("start")
img = cv2.imread('imgs/img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

height, width, channels = img.shape

cv2.namedWindow('image')
cv2.createTrackbar('min', 'image', 0, 255, nothing)

cv2.imshow("image", img)

#---

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 200, 0)
thickness = 2
#---
while (1):
    a = cv2.getTrackbarPos('min', 'image')
    ret, thresh = cv2.threshold(gray, a, 255, cv2.THRESH_BINARY)

    count = cv2.countNonZero(thresh)
    color_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    falcon = cv2.multiply(cv2.multiply(color_img, 1/255), img)

    thresh = cv2.putText(color_img, 'count '+str(count/(width*height)), org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("output", color_img)
    cv2.imshow("output_rgb", falcon)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
