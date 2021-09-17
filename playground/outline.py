import cv2
import math
import numpy as np

print("start")
img = cv2.imread('../imgs/cat.jpg')

red_noise = cv2.GaussianBlur(img, (3,3), 0,0, cv2.BORDER_DEFAULT)
gray = cv2.cvtColor(red_noise, cv2.COLOR_RGB2GRAY)

grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, -30)

#canny = cv2.Canny(img, 100, 120)

cv2.imshow("image", img)
cv2.imshow("sobel_x", grad_x)
cv2.imshow("sobel_abs_x", abs_grad_x)


while (1):

    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break