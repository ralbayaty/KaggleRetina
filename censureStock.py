'''
Initialization:
	min_scale : int, optional
		Minimum scale to extract keypoints from.

	max_scale : int, optional
		Maximum scale to extract keypoints from. The keypoints will be extracted from all the scales except the first and the last i.e. from the scales in the range [min_scale + 1, max_scale - 1]. The filter sizes for different scales is such that the two adjacent scales comprise of an octave.

	mode : {‘DoB’, ‘Octagon’, ‘STAR’}, optional
		Type of bi-level filter used to get the scales of the input image. Possible values are ‘DoB’, ‘Octagon’ and ‘STAR’. The three modes represent the shape of the bi-level filters i.e. box(square), octagon and star respectively. For instance, a bi-level octagon filter consists of a smaller inner octagon and a larger outer octagon with the filter weights being uniformly negative in both the inner octagon while uniformly positive in the difference region. Use STAR and Octagon for better features and DoB for better performance.

	non_max_threshold : float, optional
		Threshold value used to suppress maximas and minimas with a weak magnitude response obtained after Non-Maximal Suppression.

	line_threshold : float, optional
		Threshold for rejecting interest points which have ratio of principal curvatures greater than this value.


Attributes:
	keypoints:	((N, 2) array) 
		Keypoint coordinates as (row, col).
	scales:	((N, ) array)
		Corresponding scales.

'''


from skimage import data
from skimage import transform as tf
from skimage.feature import CENSURE
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


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