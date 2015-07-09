import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
from skimage import morphology


#########

try:
    file_name = sys.argv[1]
except:
    print("Didn't give me a file...")
    file_name = "Lenna.png"


img = cv2.imread(file_name,1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray_thres = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
skel = morphology.skeletonize(gray_thres > 0)

plt.subplot(121), plt.imshow(gray_thres, 'gray'), plt.axis("off")
plt.subplot(122), plt.imshow(skel, 'gray'), plt.axis("off")
plt.show()
