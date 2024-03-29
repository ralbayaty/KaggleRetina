from skimage import data
from skimage import transform as tf
from skimage.feature import CENSURE
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)


img1 = rgb2gray(data.astronaut())
tform = tf.AffineTransform(scale=(1.5, 1.5), rotation=0.5,
                           translation=(150, -200))
img2 = tf.warp(img1, tform)

detector = CENSURE()

# Defaults: min_scale=1, max_scale=7, mode='DoB', non_max_threshold=0.15, line_threshold=10

fig, ax = plt.subplots(nrows=1, ncols=2)

plt.gray()

detector.detect(img1)

ax[0].imshow(img1)
ax[0].axis('off')
ax[0].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              2 ** detector.scales, facecolors='none', edgecolors='r')

detector.detect(img2)

ax[1].imshow(img2)
ax[1].axis('off')
ax[1].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              2 ** detector.scales, facecolors='none', edgecolors='r')

plt.show()