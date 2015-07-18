import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv
import matplotlib.pyplot as plt
import pylab

def gaussian_filter(kernel_shape):
    x = np.zeros((kernel_shape, kernel_shape), dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = np.floor(kernel_shape / 2.)
    for i in xrange(0, kernel_shape):
        for j in xrange(0, kernel_shape):
            x[i, j] = gauss(i - mid, j - mid)

    return x / np.sum(x)


def lecun_lcn(input, img_shape, kernel_shape, threshold=0.0001):
    input = input.reshape(input.shape[0], 1, img_shape[0], img_shape[1])
    X = T.matrix(dtype=theano.config.floatX)
    X = X.reshape(input.shape)

    filter_shape = (1, 1, kernel_shape, kernel_shape)
    filters = gaussian_filter(kernel_shape).reshape(filter_shape)

    convout = conv.conv2d(input=X,
                          filters=filters,
                          image_shape=(input.shape[0], 1, img_shape[0], img_shape[1]),
                          filter_shape=filter_shape,
                          border_mode='full')

    # For each pixel, remove mean of 9x9 neighborhood
    mid = int(np.floor(kernel_shape / 2.))
    centered_X = X - convout[:, :, mid:-mid, mid:-mid]
    centered_X = X - convout[:, :, mid:-mid, mid:-mid]

    # Scale down norm of 9x9 patch if norm is bigger than 1
    sum_sqr_XX = conv.conv2d(input=centered_X ** 2,
                             filters=filters,
                             image_shape=(input.shape[0], 1, img_shape[0], img_shape[1]),
                             filter_shape=filter_shape,
                             border_mode='full')

    denom = T.sqrt(sum_sqr_XX[:, :, mid:-mid, mid:-mid])
    per_img_mean = denom.mean(axis=[1, 2])
    divisor = T.largest(per_img_mean.dimshuffle(0, 'x', 'x', 1), denom)
    divisor = T.maximum(divisor, threshold)

    new_X = centered_X / divisor
    new_X = new_X.dimshuffle(0, 2, 3, 1)
    new_X = new_X.flatten(ndim=3)

    f = theano.function([X], new_X)
    return f(input)

if __name__=='__main__':
    # x_img = plt.imread("./Lenna.png") #change as needed
    x_img = plt.imread("sample/10_left.jpeg")

    x_img = x_img.reshape(1, x_img.shape[0], x_img.shape[1], x_img.shape[2])
    for d in range(3):
            x_img[:, :, :, d] = lecun_lcn(x_img[:, :, :, d], (x_img.shape[1], x_img.shape[2]), 9)
    x_img = x_img[0]

    pylab.gray()
    pylab.axis('off'); pylab.imshow(x_img)
    pylab.show()