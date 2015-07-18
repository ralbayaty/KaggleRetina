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
    files = []
    file_names = []
    file_names.append('sample/10_left.jpeg')
    file_names.append('sample/10_right.jpeg')
    file_names.append('sample/13_left.jpeg')
    file_names.append('sample/13_right.jpeg')
    file_names.append('sample/15_left.jpeg')
    file_names.append('sample/15_right.jpeg')
    file_names.append('sample/16_left.jpeg')
    file_names.append('sample/16_right.jpeg')
    file_names.append('sample/17_left.jpeg')
    file_names.append('sample/17_right.jpeg')

    for i in range(10):
      files.append(cv2.imread(file_names[i],1))

    len(files)
    raw_input("")

img1 = cv2.imread(str(file1),1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread(str(file2),1)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Find keypoints and desccriptors using SIFT and RootSIFT
sift = cv2.xfeatures2d.SIFT_create() 
kp1, des1 = sift.detectAndCompute(gray1,None)
des_root1 = rootsift(kp1, des1)
kp2, des2 = sift.detectAndCompute(gray2,None)
des_root2 = rootsift(kp2, des2)

print("Image 1:")
print("RootSIFT: kps=%d, descriptors=%s " % (len(kp1), des_root1.shape))
print("\nImage 2:")
print("RootSIFT: kps=%d, descriptors=%s " % (len(kp2), des_root2.shape))

temp1, temp2 = compare_features(des_root1,des_root2,kind='euclid')
print("RootSIFT:RootSIFT    " + str(temp1))


# BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des_root1,des_root2, k=2)

# FLANNMatcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

MIN_MATCH_COUNT = 10

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3), plt.axis('off'), plt.show()