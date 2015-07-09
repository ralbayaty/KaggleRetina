#!/usr/bin/python

'''
This example illustrates how to use cv2.HoughCircles() function.
Usage: ./houghcircles.py [<image_name>]
image argument defaults to ../data/board.jpg
'''

import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt


print __doc__
try:
    fn = sys.argv[1]
except:
    fn = "../data/board.jpg"

src = cv2.imread(fn, 1)
img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img, 5)
cimg = src.copy() # numpy function

# cv2.HoughCircles(image, method, dp, minDist, circles, param1, param2, minRadius, maxRadius)
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 500, np.array([]), 100, 30, 100, 5000)
a, b, c = circles.shape
print(circles[:,:,0], circles[:,:,1], circles[:,:,2])
for i in range(b):
    cv2.circle(cimg, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(cimg, (circles[0][i][0], circles[0][i][1]), 2, (0, 255, 0), 3, cv2.LINE_AA) # draw center of circle

# cv2.imshow("source", src)
# cv2.imshow("detected circles", cimg)
# cv2.waitKey(0)

plt.subplot(121), plt.imshow(src, 'gray')
plt.subplot(122), plt.imshow(cimg,'gray')
plt.show()
