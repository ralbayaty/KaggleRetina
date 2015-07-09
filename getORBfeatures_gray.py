from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys
import os
import numpy as np
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
    folder1 = sys.argv[1]
    if folder1 not in ["sample", "train", "test"]:
      print("The folder name provided wasn't: sample, train, or test; using sample folder.")
      folder1  = "sample"
except:
    print("Didn't give me a folder; using sample folder.")
    folder1 = "sample"
issues = []
# file_names = os.listdir("/home/dick/Documents/Kaggle/" + folder1)
# file_names = os.listdir("/media/dick/Storage64GB_2/" + folder1)
file_names = os.listdir("/media/dick/External/KaggleRetina/" + folder1)
N = len(file_names)
print(str(N) + ' files')
print("Progress: starting...")
for i in range(N):
    # Read image from file, then inspect the image dimensions
    img = cv2.imread("/media/dick/External/KaggleRetina/" + folder1 + "/" + file_names[i],1)
    print(file_names[i]),
    height, width, channels = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    del img
    # Make a PIL image so we can use PIL.Image.thumbnail to resize if needed
    gray_ = Image.fromarray(gray)

    # Check if dimensions are above desired, if so then resize keepig aspect ratio
    m, n = 512,512
    if height > m or width > n:
        gray_.thumbnail((m,n), Image.ANTIALIAS)

    orb = ORB(n_keypoints=100)

    try:
        orb.detect_and_extract(gray_)
    except IndexError:
        print(file_names[i] + " had an issue.")
        issues.append(file_names[i])
        continue
    kp = orb.keypoints
    des = orb.descriptors
    print(len(des))

    #Store keypoint features 
    temp_array = []
    temp = pickle_keypoints(kp, des) 
    temp_array.append(temp)
    pickle.dump(temp_array, open("features/" + folder1 + "/orb/gray/" + file_names[i][:-5] + "_orb.pkl", "wb"))

    # temp = str(float((i+1)*100/N))
    print("Progress: " + str(i) + "/" + str(N))
f = open('issues_test.txt', 'w')
simplejson.dump(issues, f)
f.close()