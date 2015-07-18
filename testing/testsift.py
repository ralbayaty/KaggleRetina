import cv2
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

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.144]).astype('uint8')

def hellinger_kernel(x,y):
  return np.sum(np.dot(np.sqrt(x), np.sqrt(y)))

def compare_features(x,y, eps=1e-1):
  # Ensure the two features are the same size
  m1,n1 = x.shape
  m2,n2 = y.shape
  sim = 0
  count = 0
  for i in range(m1):
    for j in range(m2):
      if j >= i:
        count += 1
        if np.linalg.norm(np.array(np.subtract(x[i], y[j]))) <= eps:
          sim += 1
      
  return float(sim)/count

#########

img = cv2.imread('Lenna.png',1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

img2 = cv2.imread('sample/10_left.jpeg',1)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
gray2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)

# Find keypoints and desccriptors using SIFT
sift = cv2.xfeatures2d.SIFT_create() 
kp, des = sift.detectAndCompute(gray,None)

img_s=cv2.drawKeypoints(gray,kp,gray)
img_s_rich=cv2.drawKeypoints(gray,kp,gray,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Find keypoints and descriptors using RootSIFT
des_root = rootsift(kp, des)

print("Image 1 SIFT:     kps=%d, descriptors=%s " % (len(kp), des.shape))

kp2, des2 = sift.detectAndCompute(gray2,None)
des_root2 = rootsift(kp2, des2)

print("Image 2 SIFT:     kps=%d, descriptors=%s " % (len(kp2), des2.shape))

img2_s=cv2.drawKeypoints(gray2,kp2,gray2)
img2_s_rich=cv2.drawKeypoints(gray2,kp2,gray2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


plt.subplot(231), plt.imshow(img), plt.axis("off")
plt.subplot(232), plt.imshow(img_s), plt.axis("off")
plt.subplot(233), plt.imshow(img_s_rich), plt.axis("off")
plt.subplot(235), plt.imshow(img2_s), plt.axis("off")
plt.subplot(236), plt.imshow(img2_s_rich), plt.axis("off")
plt.show()


# print(compare_features(des,des))
# print(compare_features(des,des_root))
# print(compare_features(des_root,des_root))

# print(compare_features(des2,des2))
# print(compare_features(des2,des_root2))
# print(compare_features(des_root2,des_root2))

# print(compare_features(des,des2))
# print(compare_features(des,des_root2))
# print(compare_features(des_root,des_root2))

# cv2.imwrite('sift_keypoints.png',img)
# cv2.imwrite('sift_keypoints2.png',img)




# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des,des2,k=3)
print(len(matches))

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
  print(m)
  print(n)
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
a = len(matches[0])
for i in range(len(matches)):
  for m in range(a):
    for n in range(a):
      if matches[i][m].distance < 0.7*matches[i][n].distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img,kp,img2,kp2,matches,None,**draw_params)

plt.imshow(img3), plt.axis('off')
plt.show()