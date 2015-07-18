import cv2
import numpy as np
from matplotlib import pyplot as plt


def cmask((nx, ny),(a,b),radius):
	# Square
	# mask = np.zeros((nx,ny), np.uint8)
	# a = m/2 - 1
	# b = n/2 - 1
	# mask[a-radius:b-radius, a+radius:b+radius] = 255
	# Circle
	y,x = np.ogrid[-a:nx-a,-b:ny-b]
	mask = np.zeros((nx,ny), np.uint8)
	mask = x*x + y*y <= radius*radius

	return 255*mask.astype('uint8')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144]).astype('uint8')


##################
# Plot color histograms
# img = cv2.imread('sample/10_left.jpeg')
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     histr2 = histr

#     cv2.normalize(histr, histr2, alpha = 1,norm_type = cv2.NORM_L2)
#     histr[0], histr[1] = 0, 0
#     histr2[0], histr2[1] = 0, 0
#     plt.subplot(1,2,1), plt.plot(histr,color = col), plt.xlim([0,256])
#     plt.subplot(1,2,2), plt.plot(histr2,color = col), plt.xlim([0,256])

# plt.show()

img = cv2.imread('sample/10_left.jpeg',1)
img_gray = rgb2gray(img)

# Create a mask
m, n = img.shape[:2]
a, b = m/2 - 1, n/2 - 1
mask = cmask((m,n), (a,b), 1500)
masked_img = cv2.bitwise_and(img_gray, img_gray, mask = mask)

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img_gray],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img_gray],[0],mask,[256],[0,256])

# Normalize the histograms to: [0,1]
# hist_full = cv2.normalize(hist_full, hist_full, alpha = 1, norm_type = cv2.NORM_L2)
# hist_mask = cv2.normalize(hist_full, hist_full, alpha = 1, norm_type = cv2.NORM_L2)
hist_full = np.divide(hist_full, np.sum(hist_full))
hist_mask = np.divide(hist_mask, np.sum(hist_mask))
print(hist_full[0], sum(hist_full[1:]))
print(np.sum(hist_full))

# Remove the black pixel counts that are not within the mask
# hist_mask[0] = hist_mask[0] - (m*n - np.sum(np.divide(mask,255).astype('uint8')))
# hist_full[0] = hist_full[0] - (m*n - np.sum(np.divide(mask,255).astype('uint8')))

# Clip the saturation at 0 and 255
hist_mask[0], hist_mask[255] = 0, 0
hist_full[0], hist_full[255] = 0, 0
plt.subplot(231), plt.imshow(img_gray, 'gray'), plt.axis('off'), plt.title('image')
plt.subplot(232), plt.imshow(mask, 'gray'), plt.axis('off'), plt.title('filter mask')
plt.subplot(233), plt.imshow(masked_img, 'gray'), plt.axis('off'), plt.title('image with mask')
plt.subplot(234), plt.plot(hist_full), plt.plot(hist_mask), plt.title('BW Histograms')
plt.xlim([0,256])
# plt.show()

# Plot the RGB histograms
hist_R = cv2.calcHist([img[:,:,0]],[0],mask,[256],[0,256])
hist_G = cv2.calcHist([img[:,:,1]],[0],mask,[256],[0,256])
hist_B = cv2.calcHist([img[:,:,2]],[0],mask,[256],[0,256])

hist_R = cv2.normalize(hist_R, hist_R, alpha = 1,norm_type = cv2.NORM_L2)
hist_G = cv2.normalize(hist_G, hist_G, alpha = 1,norm_type = cv2.NORM_L2)
hist_B = cv2.normalize(hist_B, hist_B, alpha = 1,norm_type = cv2.NORM_L2)

plt.subplot(235), plt.plot(hist_R, 'red'), plt.plot(hist_G, 'green'), plt.plot(hist_B, 'blue') 
plt.title('RGB Histograms'), plt.xlim([0,256])
plt.show()