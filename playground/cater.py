import cv2
import math
import numpy as np

print("start")

img = cv2.imread('../imgs/cat.png')
img = cv2.resize(img, (500, 600), interpolation=cv2.INTER_AREA)
src = img.copy()

cv2.namedWindow('image')

tool = 1

drawing = False
pt1_x, pt1_y = None, None


def line_drawing(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    if tool == 1:
        if event == cv2.EVENT_LBUTTONDOWN:
            pt1_x, pt1_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.line(img, (pt1_x, pt1_y), (x, y), color=(10, 120, 255), thickness=3)
                pt1_x, pt1_y = x, y

    if tool == 2:
        if event == cv2.EVENT_LBUTTONDOWN:
            pt1_x, pt1_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            end_x, end_y = x, y
            radius = math.sqrt((end_x - pt1_x) ** 2 + (end_y - pt1_y) ** 2)
            center = (pt1_x, pt1_y)
            cv2.cv2.circle(img, center, int(radius * 0.5), (200, 200, 100), 2)
        pass
    if tool == 3:
        if event == cv2.EVENT_LBUTTONDOWN:
            pt1_x, pt1_y = x, y
            print(str(pt1_x) + ", " + str(pt1_y))


cv2.setMouseCallback('image', line_drawing)

cv2.imshow("src", src)

cv2.cv2.circle(img, (179, 269), int(40 * 0.5), (20, 250, 200), 2)
cv2.cv2.circle(img, (336, 273), int(40 * 0.5), (20, 250, 200), 2)

cv2.line(img, (137, 280), (170, 237), color=(10, 120, 255), thickness=3)
cv2.line(img, (170, 237), (222, 277), color=(10, 120, 255), thickness=3)
cv2.line(img, (222, 277), (173, 308), color=(10, 120, 255), thickness=3)
cv2.line(img, (173, 308), (137, 280), color=(10, 120, 255), thickness=3)

cv2.line(img, (296, 279), (337, 243), color=(10, 120, 255), thickness=3)
cv2.line(img, (337, 243), (379, 276), color=(10, 120, 255), thickness=3)
cv2.line(img, (379, 276), (339, 310), color=(10, 120, 255), thickness=3)
cv2.line(img, (339, 310), (296, 279), color=(10, 120, 255), thickness=3)

cv2.line(img, (230, 336), (257, 362), color=(10, 220, 255), thickness=3)
cv2.line(img, (257, 362), (288, 335), color=(10, 220, 255), thickness=3)
cv2.line(img, (288, 335), (230, 336), color=(10, 220, 255), thickness=3)

cv2.ellipse(img, (179, 269), (20, 20), 0, 0, 360, (0, 0, 255), 5)

cv2.ellipse(img, (102, 102), (30, 70), -20, 0, 360, (0, 0, 255), 5)
cv2.ellipse(img, (418, 117), (30, 70), 20, 0, 120, (0, 0, 255), 5)
cv2.ellipse(img, (418, 117), (30, 70), 20, 120, 240, (250, 0, 1), 5)
cv2.ellipse(img, (418, 117), (30, 70), 20, 240, 360, (250, 0, 250), 5)

while (1):

    cv2.imshow("image", img)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    if k == ord('q'):
        print("tool 1")
        tool = 1
        drawing = False
    if k == ord('w'):
        print("tool 2")
        tool = 2
        drawing = False
    if k == ord('e'):
        print("tool 3")
        tool = 3
        drawing = False

    if k == ord('c'):
        print("clear")
        img = src.copy()
    if k == ord('s'):
        cv2.imwrite("../imgs/out.png", img)
