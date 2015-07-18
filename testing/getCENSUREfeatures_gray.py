from skimage.feature import CENSURE
from skimage.color import rgb2gray
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys
import os
import numpy as np
import cPickle as pickle


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

min_scale = 1
max_scale = 7
mode = "STAR" 
non_max_threshold = 0.006
line_threshold = 10


file_names = os.listdir("/home/dick/Documents/Kaggle/" + folder1)
N = len(file_names)
print(str(N) + ' files')
print("Progress: starting...")
for i in range(N):
    # Read image from file, then inspect the image dimensions
    img = cv2.imread(folder1 + "/" + file_names[i],1)
    height, width, channels = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    del img
    # Make a PIL image so we can use PIL.Image.thumbnail to resize if needed
    gray_ = Image.fromarray(gray)

    # Check if dimensions are above desired, if so then resize keepig aspect ratio
    m, n = 512, 512
    if height > m or width > n:
        gray_.thumbnail((m,n), Image.ANTIALIAS)

    censure = CENSURE(min_scale=min_scale, max_scale=max_scale, mode=mode, 
                            non_max_threshold=non_max_threshold, line_threshold=line_threshold)

    censure.detect(gray_)
    kp = censure.keypoints
    scales = censure.scales
    # print(len(scales))

    #Store keypoint features 
    temp_array = []
    temp = pickle_keypoints(kp, scales) 
    temp_array.append(temp)
    pickle.dump(temp_array, open("features/" + folder1 + "/censure/gray/" + file_names[i][:-5] + "_censure.pkl", "wb"))

    temp = str(float((i+1)*100/N))
    print("Progress: " + temp + " %")