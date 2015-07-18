import os

from sklearn import neighbors
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA

import make_dataset

if __name__ == '__main__':

    test_file_names = os.listdir('/media/dick/Storage/test')
    N_test = len(test_file_names)
    print(str(N_test) + ' test files.')

    train_file_names = os.listdir('/media/dick/Storage64GB_2/train')
    N_train = len(train_file_names)
    print(str(N_train) + ' train files.')

    array_train, array_list_train = make_dataset.read_retina_images(which_set='train')
    array_test, array_list_test = make_dataset.read_retina_images(which_set='test')

    # train_labels = np.load('/media/dick/Storage64GB_2/train_labels.npy')
    # print(train_labels.shape)

    train_imgs, train_labels = make_dataset.read_retina_images(which_set='train')
    test_imgs, test_labels = make_dataset.read_retina_images(which_set='test')
    N_train, m_train, n_train, chan_train = train_imgs.shape
    N_test, m_test, n_test, chan_test = test_imgs.shape

    # cv2.imshow('image', train_imgs[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(train_imgs.reshape((N_train, m_train*n_train))[:5000].shape)

    # Using sklearn
    NN = neighbors.NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(train_imgs.reshape((N_train, m_train*n_train))[:1000])

    # Using cv2
    # knn = cv2.KNearest()
    # knn.train(train_imgs.reshape((N_train, m_train*n_train))[:1000], train_labels[:1000])


    # distances, indices = NN.kneighbors(train_imgs.reshape((N_train, m_train*n_train))[:1])
    # print(indices)
    # print(distances)
    # print(train_labels[indices[0]])
    #
    # distances, indices = NN.kneighbors(test_imgs.reshape((N_test, m_test*n_test))[:1])
    # print(indices)
    # print(distances)
    # print(train_labels[indices[0]])

    # plt.imshow(train_imgs[0].reshape((200, 200)), cmap='gray', interpolation='bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()