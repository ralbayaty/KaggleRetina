import itertools
import numpy as np
import os
import csv
from PIL import Image
from pylearn2.space import VectorSpace, CompositeSpace, Conv2DSpace, IndexSpace
from pylearn2.datasets import preprocessing
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import cache
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng
import cv2
from math import floor


def write_retina_images(file_list, which_set='train', dtype=None, ow=True):
    if which_set not in ['train', 'test']:
        raise ValueError("Need to pass either 'train' or 'test'.")
    names, labels = read_retina_train_labels()
    N = 45
    total_num = N*len(file_list)
    w1 = 201
    w2 = 201
    if which_set is 'test':
        im_path = '/media/dick/Storage1TB/test/'
        array = np.memmap('/media/dick/External/KaggleRetina/test.mmap', 
                   mode='w+', shape=(total_num, w1-1, w2-1, 1))
        array_labels = np.memmap('/media/dick/External/KaggleRetina/test_labels.mmap', 
                   mode='w+', shape=(total_num, 1), dtype='S12')
        issues_file = open('/media/dick/External/KaggleRetina/test_issues.txt', 'w')
    if which_set is 'train':
        im_path = '/media/dick/Storage64GB_2/train/'
        array = np.memmap('/media/dick/External/KaggleRetina/train.mmap', 
                   mode='w+', shape=(total_num, w1-1, w2-1, 1))
        array_labels = np.memmap('/media/dick/External/KaggleRetina/train_labels.mmap', 
                   mode='w+', shape=(total_num, 1))
        issues_file = open('/media/dick/External/KaggleRetina/train_issues.txt', 'w')
    if ow is True:
        progress = 0.0
        for file_name in file_list:
            img = cv2.imread(im_path + file_name, 1)
            img = np.asarray(img)
            img_patches = make_patches(img, N=N, w1=w1, w2=w2)
            m, n, channel = img.shape
            print(m, n, channel)
            if m >= w1 and n >= w2: 
                for i in range(N):
                    array[i,:,:] = np.reshape(img_patches[i], (w1-1,w2-1,1))
                    # print(file_name[:-5], names[names.index(file_name[:-5])], labels[names.index(file_name[:-5])])
                    if which_set is 'train':
                        array_labels[i] = labels[names.index(file_name[:-5])]
                    else:
                        array_labels[i] = [file_name[:-5]]
                    progress += 1
                print("\rProgress: " + str(round(progress*100/total_num, 3)) + " %"),
            else:
                issues_file.write(file_name)
        issues_file.close()
        if dtype:
            dtype = numpy.dtype(dtype)
            # If the user wants booleans, threshold at half the range.
            if dtype.kind is 'b':
                array = array >= 128
            else:
                # Otherwise, just convert.
                array = array.astype(dtype)
            # I don't know why you'd ever turn MNIST into complex,
            # but just in case, check for float *or* complex dtypes.
            # Either way, map to the unit interval.
            if dtype.kind in ('f', 'c'):
                array /= 255.
    else:
        pass
    return array, array_labels


def write_retina_labels(file_list):
    names, labels = read_retina_train_labels()
    N = 45
    total_num = N*len(file_list)
    w1 = 201
    w2 = 201

    im_path = '/media/dick/Storage64GB_2/train/'
    array_labels = np.memmap('/media/dick/External/KaggleRetina/train_labels.mmap', 
                             mode='w+', shape=(total_num, 1))
    progress = 0.0
    for file_name in file_list:
        # img = cv2.imread(im_path + file_name, 1)
        # img = np.asarray(img)
        # m, n, channel = img.shape
        # if m >= w1 and n >= w2: 
            for i in range(N):
                array_labels[int(progress)] = labels[names.index(file_name[:-5])]
                progress += 1
            # print(array_labels[int(progress)])
            print("\rProgress: " + str(progress) + '/'+ str(total_num) + ' , ' + str(round(progress*100/total_num, 3)) + " %"),
        # else:
        #     issues_file.write(file_name)
    return array_labels


def read_retina_images(which_set='train'):

    if which_set not in ['train', 'test']:
        raise ValueError("Need to pass either 'train' or 'test'.")

    N = 45
    w1 = 201
    w2 = 201

    if which_set == 'test':
        im_path = '/media/dick/Storage1TB/test/'
        file_list = os.listdir(im_path)
        total_num = N*len(file_list)
        array = np.memmap('/media/dick/External/KaggleRetina/test.mmap', 
                   mode='r', shape=(total_num, w1-1, w2-1, 1))
        array_labels = np.memmap('/media/dick/External/KaggleRetina/test_labels.mmap', 
                   mode='r', shape=(total_num, 1), dtype='S12')
        # issues_file = open('/media/dick/External/KaggleRetina/test_issues.txt', 'w')
    if which_set == 'train':
        im_path = '/media/dick/Storage64GB_2/train/'
        file_list = os.listdir(im_path)
        total_num = N*len(file_list)
        array = np.memmap('/media/dick/External/KaggleRetina/train.mmap', 
                   mode='r', shape=(total_num, w1-1, w2-1, 1))
        array_labels = np.memmap('/media/dick/External/KaggleRetina/train_labels.mmap', 
                   mode='r', shape=(total_num, 1))
        # issues_file = open('/media/dick/External/KaggleRetina/train_issues.txt', 'w')

    return array, array_labels


def make_patches(img, N=100, black_thres=0.5, w1=201, w2=201):
    """
    Takes an image and returns a user selected set of random patches.

    Inputs:
    ----------
    img : 2D array (potentially with color channels)
        An image loaded from cv2 or PIL
    N : int
        The number of desired patches.
    black_thres: float
        The threshold for how much black is allowed to be in a patch
    w1 and w2: int
        The size of the patches.

    Outputs:
    -------
    patches : ndarray, shape (n_patches, n_rows, n_cols)
        An image array, with individual examples indexed along the
        first axis and the image dimensions along the second and
        third axis.

    Notes
    -----

    """

    m, n, channels = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_ = Image.fromarray(gray)

    # if m > w1 or n > w2:
    #     gray_.thumbnail((m,n), Image.ANTIALIAS)

    # Need to add 1 to each if w1 or w2 is even
    center1 = int(floor(w1/2))
    center2 = int(floor(w2/2))
    stop1 = m - center1 
    stop2 = n - center2
    if stop1 <= center1:
        # raise ValueError("Stop1 was smaller than center1.")
        return
    if stop2 <= center2:
        # raise ValueError("Stop2 was smaller than center2.")
        return
    patches = []

    while len(patches) < N:
        # select a random center location for the patch from the image
        rand_m = np.random.randint(center1, stop1)
        rand_n = np.random.randint(center2, stop2)

        # Ensure random selected pixel locations are valid
        assert rand_m-center1 >= 0
        assert rand_m+center1 <= m
        assert rand_n-center2 >= 0
        assert rand_n+center2 <= n

        patch = np.copy(gray[(rand_m-center1):(rand_m+center1), (rand_n-center2):(rand_n+center2)])

        hist_full = cv2.calcHist([patch], [0], None, [256], [0, 256])
        if sum(hist_full) > 0:
            hist_full = np.divide(hist_full, sum(hist_full))
            if hist_full[0] < black_thres:
                patches.append(patch)
    return patches


def read_retina_train_labels():
    with open('/media/dick/Storage64GB_2/trainLabels.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        names = []
        labels = []
        header_flag = True
        for row in reader:
            if header_flag is False:
                names.append(row[0])
                labels.append(row[1])
            else:
                header_flag = False

    return names, labels


def generate(which_set='train', start=None, stop=None, 
             data_type='vector', norm_data=False,
             save_data=False, save_path=None):
    """

    Parameters
    ----------
    which_set: str
        Two options currently: 'train' or 'test'
    start: int
        The starting point of the 'train', 'valid', or 'test' dataset.
    stop: int
        The stopping point of the 'train', 'valid', or 'test' dataset.
    data_type: str
        Two options: 'conv2d' or 'vector' representing either a Conv2DSpace or VectorSpace for the data
    norm_data: bool
        Normalize the datasets?
    save_data: bool
        Save the generated dataset to disk?
    save_path: str
        Desired location to save dataset.
    """

    if which_set == 'train':
        im_path = '/media/dick/External/KaggleRetina/'
        label_path = '/media/dick/External/KaggleRetina/'
        # im_path = '/media/dick/Storage64GB_2/train'
        # label_path = '/media/dick/Storage64GB_2'
    else:
        assert which_set == 'test'
        im_path = '/media/dick/External/KaggleRetina/'
        # im_path = '/media/dick/Storage/test'

    if data_type not in ['vector', 'conv2d']:
        print(data_type)
        raise ValueError('Data type was an incorrect value. Can be either "vector" or "conv2d".')

    retina_imgs, retina_labels = read_retina_images(which_set)
    y = np.load(which_set + '_labels.npy')

    N, m, n, chan = retina_imgs.shape
    print("N is " + str(N) + ", m is " + str(m) + ", n is " + str(n) + ", channels is " + str(chan))
    # retina_imgs = retina_imgs.reshape(N, m, n, 1)

    view_converter = dense_design_matrix.DefaultViewConverter((m, n, 1))

    design_X = view_converter.topo_view_to_design_mat(retina_imgs)

    # Normalize data:
    if norm_data is True:
        pipeline = preprocessing.Pipeline()
        gcn = preprocessing.GlobalContrastNormalization(
            sqrt_bias=10., use_std=True)
        pipeline.items.append(gcn)
        X_ImP = dense_design_matrix.DenseDesignMatrix(X=X)
        X_ImP.apply_preprocessor(preprocessor=pipeline, can_fit=True)

        X = X_ImP.X
    else:
        X = design_X

    # As a VectorSpace
    if data_type == 'vector':
        data_specs = (CompositeSpace([ VectorSpace(m * n),
                                       IndexSpace(dim=1,max_labels=5) 
                                     ]),
                      ('features', 'targets'))
        train = VectorSpacesDataset(data=(X, y), data_specs=data_specs)

    # As a Conv2DSpace
    if data_type == 'conv2d':
        topo_X = view_converter.design_mat_to_topo_view(X)
        axes = ('b', 0, 1, 'c')
        data_specs = (CompositeSpace([ Conv2DSpace((m, n), num_channels=chan, axes=axes), 
                                       IndexSpace(dim=1,max_labels=5) 
                                     ]),
                      ('features', 'targets'))
        train = VectorSpacesDataset((topo_X, y), data_specs=data_specs)

    if save_data is True:
        print("Saving dataset...")
        save_path = os.path.dirname(os.path.realpath(__file__))
        serial.save(os.path.join(save_path, 'dataset.pkl'), train)

    return train

if __name__ == '__main__':
    
    test_file_names = os.listdir('/media/dick/Storage/test')
    N_test = len(test_file_names)
    print(str(N_test) + ' test files.')

    train_file_names = os.listdir('/media/dick/Storage64GB_2/train')
    N_train = len(train_file_names)
    print(str(N_train) + ' train files.')

    # write_retina_images(train_file_names, which_set='train')
    # write_retina_images(test_file_names, which_set='test')

    array_train, array_list_train = read_retina_images(which_set='train')
    array_test, array_list_test = read_retina_images(which_set='test')


    # train_labels = write_retina_labels(train_file_names)
    # np.save('train_labels.npy', train_labels)
    train_labels = np.load('/media/dick/Storage64GB_2/train_labels.npy')
    print(train_labels.shape)

