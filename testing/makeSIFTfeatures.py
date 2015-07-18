import cv2
import sys
import os
import numpy as np
import cPickle as pickle


# RootSIFT
def rootsift(kps, des, eps=1e-7):
  # If there are no keypoints or descriptors, return an empty tuple
  if len(kps) == 0:
    return ([], None)

  # Apply the Hellinger kernel by: 1) L1-normalizing, 2) square-root
  des /= (des.sum(axis=1, keepdims=True) + eps)
  des = np.sqrt(des)

  # Unsure if L2-normalizing after last step is necessary
  #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
  
  # Return a tuple of the keypoints and descriptors
  return des


def pickle_keypoints(keypoints, descriptors): 
  i = 0 
  temp_array = [] 
  for point in keypoints: 
    temp = (point.pt, point.size, point.angle, point.response, point.octave, 
    point.class_id, descriptors[i]) 
    ++i 
    temp_array.append(temp) 
    return temp_array 

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
print("Progress: 0 %")
for i in range(N):
  img = cv2.imread(folder1 + "/" + file_names[i],1)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Find keypoints and desccriptors using SIFT and RootSIFT
  sift = cv2.xfeatures2d.SIFT_create() 
  kp, des = sift.detectAndCompute(gray,None)
  des = rootsift(kp, des)

  #Store and Retrieve keypoint features 
  temp_array = [] 
  temp = pickle_keypoints(kp, des) 
  temp_array.append(temp)
  pickle.dump(temp_array, open("features/" + folder1 + "/" + file_names[i][:-5] + "_sift.pkl", "wb"))

  temp = str(float((i+1)*100/N))
  print("Progress: " + temp + " %")

  # #Retrieve Keypoint Features 
  # keypoints_database = pickle.load( open( "keypoints_database.pkl", "rb" ) ) 
  # kp1, desc1 = unpickle_keypoints(keypoints_database[0]) 
  # kp2, desc2 = unpickle_keypoints(keypoints_database[1])