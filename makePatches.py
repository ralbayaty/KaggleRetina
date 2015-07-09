from skimage.color import rgb2gray
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys
import os
import numpy as np
from math import floor
import cPickle as pickle
import simplejson


def pickle_keypoints(keypoints, descriptors): 
  temp_array = [] 
  for i in range(len(descriptors)): 
    temp = (keypoints[i], descriptors[i]) 
    temp_array.append(temp)
    return temp_array 

#########
try:
    file_name = sys.argv[1]
except:
    print("Didn't give me a file.")

img = cv2.imread(file_name, 1)
m, n, channels = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_ = Image.fromarray(gray)

# Check if dimensions are above desired, if so then resize keeping aspect ratio
w1, w2 = 201, 201
# Need to add 1 to each if w1 or w2 is even
center1 = int(floor(w1/2))
center2 = int(floor(w2/2))
black_thres = 0.50  # percentage of image that is black
N = 100
patches = []
# if m > w1 or n > w2:
#     gray_.thumbnail((m,n), Image.ANTIALIAS)

while len(patches) < N:
    # select a random center location for the patch from the image
    rand_m = np.random.randint(0+center1, m-center1)
    rand_n = np.random.randint(0+center2, n-center2)

    # Ensure random selected pixel locations are valid
    assert rand_m-center1 >= 0
    assert rand_m+center1 <= m
    assert rand_n-center2 >= 0
    assert rand_n+center2 <= n

    patch = np.copy(gray[(rand_m-center1):(rand_m+center1), (rand_n-center2):(rand_n+center2)])

    hist_full = cv2.calcHist([patch], [0], None, [256], [0, 256])
    if sum(hist_full) > 0:
        hist_full = np.divide(hist_full, sum(hist_full))
        if hist_full[0] < black_thres:
            patches.append(patch)
            cv2.imshow('patch', np.asarray(patch))
            if 0xFF & cv2.waitKey(50) == 27:
                pass
                
cv2.destroyAllWindows()

print("Finished! " + str(len(patches)) + " patches created.")