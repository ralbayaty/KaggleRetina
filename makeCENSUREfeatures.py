from skimage import data
from skimage import transform as tf
from skimage.feature import CENSURE
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage import data
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2
import sys
import os
import numpy as np
import cPickle as pickle


def pickle_keypoints(keypoints, descriptors): 
  i = 0 
  temp_array = [] 
  for point in keypoints: 
    temp = (point, descriptors[i]) 
    ++i 
    temp_array.append(temp) 
    return temp_array 

# Need to figure out how to process none cv2 kp and des
def unpickle_keypoints(array):
  keypoints = [] 
  descriptors = [] 
  for point in array: 
    temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], 
    _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5]) 
    temp_descriptor = point[6] 
    keypoints.append(temp_feature) 
    descriptors.append(temp_descriptor) 
    return keypoints, np.array(descriptors)

#########
try:
    folder1 = sys.argv[1]
    if folder1 not in ["sample", "train", "test"]:
      print("The folder name provided wasn't: sample, train, or test; using sample folder.")
      folder1  = "sample"
except:
    print("Didn't give me a folder; using sample folder.")
    folder1 = "sample"


file_names = os.listdir("/home/dick/Documents/Kaggle/" + folder1)
N = len(file_names)
print("Progress: 0 %"),
for i in range(N):
	img = cv2.imread(folder1 + "/" + file_names[i],1)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	censure = CENSURE()

	censure.detect(gray)
	kp = censure.keypoints
	scales = censure.scales
	print(len(kp))

	plt.imshow(img)
	plt.axis('off')
	plt.scatter(censure.keypoints[:, 1], censure.keypoints[:, 0],
	              2 ** censure.scales, facecolors='none', edgecolors='r')
	plt.show()

	#Store and Retrieve keypoint features 
	temp_array = []
	temp = pickle_keypoints(kp, scales) 
	temp_array.append(temp)
	pickle.dump(temp_array, open("features/" + folder1 + "/censure/gray/" + file_names[i][:-5] + "_censure.pkl", "wb"))

	temp = str(float((i+1)*100/N))
	print("\rProgress: " + temp + " %"),