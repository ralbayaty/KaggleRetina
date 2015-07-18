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

def unpickle_keypoints(array): 
    keypoints = [] 
    descriptors = [] 
    for point in array: 
        keypoints.append(point[0]) 
        descriptors.append(point[1]) 
    return keypoints, descriptors

#########
try:
    folder1 = sys.argv[1]
    if folder1 not in ["sample", "train", "test"]:
      print("The folder name provided wasn't: sample, train, or test; using sample folder.")
      folder1  = "sample"
except:
    print("Didn't give me a folder; using sample folder.")
    folder1 = "sample"

K = []
D = []

# file_names = os.listdir("/home/dick/Documents/Kaggle/" + folder1)
# file_names = os.listdir("/media/dick/Storage64GB_2/" + folder1)
file_names = os.listdir("/media/dick/External/KaggleRetina/" + folder1)
N = len(file_names)
print(str(N) + ' files')
print("Progress: starting...")
for i in range(N):

    try:
        #Retrieve Keypoint Features 
        keypoints_database = pickle.load( open("features/" + folder1 + "/orb/gray/" + file_names[i][:-5] + "_orb.pkl", "rb") ) 
        kp, des = unpickle_keypoints(keypoints_database[0])
        K.append(kp)
        D.append(np.array(des))
        print(len(D))
        print(len(D[0]))
        print(len(D[0][0]))
    except IOError:
        print("Couldn't load " + file_names[i] + " due to IOError.")
    except TypeError:
        print("Couldn't load " + file_names[i] + " due to TypeError.")


    # print("Progress: " + str(i) + "/" + str(N))
# distance = np.zeros((len(K), len(K)))

for i in range(len(K)):
    top1 = [0, 0]
    for j in range(len(K)):
        matches = match_descriptors(D[i], D[j])
        if matches.shape[1] > top1[0]:
            top1[0] = matches.shape[1]
            top1[1] = j
        # distance[i][j] = matches.shape[1]
    print(top1)
    top1 = [0, 0]
    print("Keypoints: " + str(K[i][matches[:,0]]) + ', ' + str(K[j][matches[:,1]]))
    print("Progress: " + str(i) + "/" + str(len(K)))