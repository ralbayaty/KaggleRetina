import numpy as np
import cv2
from matplotlib import pyplot as plt
from time import sleep

def patchify(img, patch_shape):
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape[:2]
    x, y = patch_shape
    shape = ((X-x+1), (Y-y+1), x, y) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
    #    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([X, 1, Y, 1])
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


file1 = 'Lenna.png'
img = cv2.cvtColor(cv2.imread(file1,1), cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

patches = patchify(img, (50,50))
contiguous_patches = np.ascontiguousarray(patches)
contiguous_patches.shape = (-1, 50**2)
print(patches.shape)
print(type(patches))
print(contiguous_patches.shape)
print(len(contiguous_patches))
print(contiguous_patches[0].reshape((50,50)).shape)

plt.imshow(patches[0][0].reshape((50,50))), plt.axis('off')
plt.show(block=True)

# for i in range(len(contiguous_patches)):
#     plt.imshow(contiguous_patches[i].reshape((50,50))), plt.axis('off')
#     plt.draw()