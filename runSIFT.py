import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt


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

def hellinger_kernel(x,y):
  return np.dot(np.sqrt(x), np.sqrt(y))

def compare_features(x, y, kind='euclid', eps=1e-1):
  # Ensure the two features are the same size
  m1,n1 = x.shape
  m2,n2 = y.shape
  d = 0
  count = 0
  counts = np.zeros((m1,m2))
  for i in range(m1):
    for j in range(m2):
      if j >= i:
        count += 1
        if kind is 'euclid':
          if np.linalg.norm(np.subtract(x[i], y[j]),2) <= eps:
            d += 1
            counts[i][j] += 1
        if kind is 'hell':
          if np.dot(np.sqrt(x[i]), np.sqrt(y[j])) <= eps:
            d += 1
            counts[i][j] += 1
      
  return float(d)/count, counts

#########

try:
    file1 = sys.argv[1]
    file2 = sys.argv[2]
except:
    file1 = 'Lenna.png'
    file2 = 'sample/10_left.jpeg'

img = cv2.imread(str(file1),1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread(str(file2),1)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Find keypoints and desccriptors using SIFT and RootSIFT
sift = cv2.xfeatures2d.SIFT_create() 
kp, des = sift.detectAndCompute(gray,None)
des_root = rootsift(kp, des)
kp2, des2 = sift.detectAndCompute(gray2,None)
des_root2 = rootsift(kp2, des2)

print("Image 1:")
print("RootSIFT: kps=%d, descriptors=%s " % (len(kp), des_root.shape))
print("\nImage 2:")
print("RootSIFT: kps=%d, descriptors=%s " % (len(kp2), des_root2.shape))

# img_sift = cv2.drawKeypoints(gray,kp,gray)
# img_sift_rich = cv2.drawKeypoints(gray,kp,gray,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# img2_sift = cv2.drawKeypoints(gray2,kp2,gray2)
# img2_sift_rich = cv2.drawKeypoints(gray2,kp2,gray2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# plt.subplot(231), plt.imshow(img), plt.axis("off")
# plt.subplot(232), plt.imshow(img_sift), plt.axis("off")
# plt.subplot(233), plt.imshow(img_sift_rich), plt.axis("off")
# plt.subplot(235), plt.imshow(img2_sift), plt.axis("off")
# plt.subplot(236), plt.imshow(img2_sift_rich), plt.axis("off")
# plt.show()


# temp1, temp2 = compare_features(des,des2,kind='hell')
# print("\nSIFT:SIFT            " + str(temp1))
# temp1, temp2 = compare_features(des,des_root2,kind='euclid')
# print("SIFT:RootSIFT        " + str(temp1))
# temp1, temp2 = compare_features(des_root,des2,kind='euclid')
# print("RootSIFT:SIFT        " + str(temp1))
temp1, temp2 = compare_features(des_root,des_root2,kind='euclid')
print("RootSIFT:RootSIFT    " + str(temp1))

## FLANN section
# # FLANN parameters
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=100)   # or pass empty dictionary
# flann = cv2.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des_root,des_root2,k=10)

# # Need to draw only good matches, so create a mask
# # matchesMask = [[0,0] for i in xrange(len(matches))]
# # print(matchesMask)
# matchesMask = []

# # ratio test as per Lowe's paper
# a = len(matches[0])
# for i in range(len(matches)):
#   for m in range(a):
#     for n in range(a):
#       # if matches[i][m].distance < 0.7*matches[i][n].distance:
#       #   matchesMask[i] = [1,0]
#       # print(matches[i][m].distance)
#       # print(matches[i][n].distance)
#       print(i,m,n)
#       temp1 = matches[i][m].distance
#       temp2 = matches[i][n].distance
#       if temp1 < 0.7*temp2:
#         matchesMask.append([1,0])
#       else:
#         matchesMask.append([0,0])
# 
# print("Made match mask.")
# 
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = 0)
# 
# img3 = cv2.drawMatchesKnn(img,kp,img2,kp2,matches,None,**draw_params)


### BF section
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_root,des_root2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print("There are " + str(len(good)) + " good matches.")

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img,kp,img2,kp2,good,None,flags=2)



plt.imshow(img3), plt.axis('off'), plt.show()