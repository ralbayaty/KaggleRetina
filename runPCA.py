import os
from sklearn.externals import joblib
import numpy as np
import csv
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn import neighbors
from collections import Counter

import make_dataset


def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def make_predictions(labels, idx, n=45):
    preds = [most_common(labels[idx[x]]) for x in range(len(idx))]
    predictions = [most_common(list(preds[x:(x+n)])) for x in range(0, len(idx), n)]
    return predictions


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
    print('# of training examples: ' + str(N_train) + '\n# of testing examples:  ' + str(N_test))

    # Using sklearn, train a model using Incremental PCA (out-of-core)
    n_components = 10
    # ipca = IncrementalPCA(n_components=n_components, batch_size=50)
    # for i in range(0, N_train, 1000):
    #     print('-> iteration: ' + str(i))
    #     ipca.partial_fit(train_imgs.reshape((N_train, m_train*n_train))[i:(i+1000)])
    #
    # # ipca.fit(train_imgs.reshape((N_train, m_train*n_train))[:50000])
    # print('Incremental PCA complete!')
    # joblib.dump(ipca, '/media/dick/External/KaggleRetina/models/ipca.pkl')
    # print('IPCA model saved.')

    # Create/Open the memory mapped variables
    X_ipca = np.memmap('/media/dick/External/KaggleRetina/train_ipca.mmap',
               mode='w+', shape=(N_train, n_components))
    X_test_ipca = np.memmap('/media/dick/External/KaggleRetina/test_ipca.mmap',
               mode='w+', shape=(N_test, n_components))
    # array_labels = np.memmap('/media/dick/External/KaggleRetina/test_labels.mmap',
    #            mode='w+', shape=(total_num, 1), dtype='S12')

    # Load the trained model
    ipca = joblib.load('/media/dick/External/KaggleRetina/models/ipca.pkl')

    # Transform the data using the model
    # N1 = 1580670
    # N2 = 2410920
    # Create in mass
    # X_ipca[:N1] = ipca.transform(train_imgs.reshape((N_train, m_train*n_train))[:N1])
    # X_test_ipca[:N2] = ipca.transform(test_imgs.reshape((N_test, m_test*n_test))[:N2])
    # Create in batch
    # print('Training data:')
    # for i in range(0, N1, 10000):
    #     print('-> iteration: ' + str(int(1+float(i)/10000)))
    #     X_ipca[i:(i+10000)] = ipca.transform(train_imgs.reshape((N_train, m_train*n_train))[i:(i+10000)])
    # print('Testing data:')
    # for i in range(0, N2, 10000):
    #     print('-> iteration: ' + str(int(1+float(i)/10000)))
    #     X_test_ipca[i:(i+10000)] = ipca.transform(test_imgs.reshape((N_test, m_test*n_test))[i:(i+10000)])
    # print('Data transformed!')

    # # Run Kernel PCA on the IPCA results
    # kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True)
    # X_kpca = kpca.fit_transform(X_ipca)
    # X_test_kpca = kpca.transform(X_test_ipca)
    # print('Kernel PCA complete!')
    # print(X_kpca.shape)

    print('Shape of training images after IPCA: ' + str(X_ipca.shape))
    print('Shape of testing images after IPCA: ' + str(X_test_ipca.shape))

    # Create the NN model
    # NN = neighbors.NearestNeighbors(n_neighbors=25, algorithm='ball_tree')
    # NN.fit(X_ipca[:N_train])
    # joblib.dump(NN, '/media/dick/External/KaggleRetina/models/NN_25.pkl')

    # Load the NN model
    NN = joblib.load('/media/dick/External/KaggleRetina/models/NN_25.pkl')

    # Load the trained model
    # NN = joblib.load('/media/dick/External/KaggleRetina/models/NN_25.pkl')

    # Test kNN on training data
    # distances, indices = NN.kneighbors(X_ipca[:10])
    # print('Indices: ' + str(indices))
    # print('Distances: ' + str(distances))
    # print(str(train_labels[indices[0]]))

    # Find kNN on test data: num_test should be a multiple of the number of sub-images per image
    num_test = 45*1000
    distances, indices = NN.kneighbors(X_test_ipca)
    joblib.dump(indices, '/media/dick/External/KaggleRetina/models/testNNindices_25.pkl')
    # print('Indices: ' + str(indices))
    # print('Distances: ' + str(distances))

    print('Number of sub-images to test: ' + str(len(indices)))
    preds = make_predictions(train_labels, indices)
    print('Number of predictions: ' + str(len(preds)))

    print('Unique labels from training images: ' + str(np.unique(train_labels)))
    print('Unique labels test images: ' + str(np.unique(np.unique(preds))))

    # Make test set labels for submission
    with open('./labels/test_labels_IPCA.csv', 'wb') as f:
        writer = csv.writer(f)
        for i in range(N_test):
            if i == 0:
                writer.writerows([['image', 'level']])
            writer.writerows([[test_file_names[i][:-5], preds[i]]])

    print("\nCSV file containing image-label pairs was created.")


    # cv2.imshow('image', train_imgs[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()