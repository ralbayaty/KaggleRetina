import os
import time
from sklearn.externals import joblib
import numpy as np
import csv
import cv2
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn import neighbors
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import entropy
from collections import Counter
from matplotlib import pyplot as plt

import make_dataset


def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def make_predictions(labels, idx, n=45):
    preds = [most_common(labels[idx[x]]) for x in range(len(idx))]
    predictions = [most_common(list(preds[x:(x+n)])) for x in range(0, len(idx), n)]
    return predictions


def find_kl(p, q):
    return entropy(p, q, base=None)


def make_test_preds(y, name):
    test_file_names = os.listdir('/media/dick/Storage1TB/test')
    N = len(test_file_names)
    # Make completely random test set labels for first submission
    with open('./labels/' + name + '.csv', 'wb') as f:
        writer = csv.writer(f)
        for i in range(len(y)):
            if i == 0:
                writer.writerows([['image', 'level']])
            writer.writerows([[test_file_names[i][:-5], int(y[i])]])

    print("\nCSV file containing image-label pairs was created.")


if __name__ == '__main__':

    test_file_names = os.listdir('/media/dick/Storage1TB/test')
    N_test = len(test_file_names)
    print(str(N_test) + ' test files.')

    train_file_names = os.listdir('/media/dick/Storage1TB/train')
    N_train = len(train_file_names)
    print(str(N_train) + ' train files.')

    # Create the image features using image patches
    # make_dataset.write_retina_images(train_file_names, which_set='train')
    # make_dataset.write_retina_images(test_file_names, which_set='test')

    # train_labels = np.load('/media/dick/Storage64GB_2/train_labels.npy')
    # print(train_labels.shape)

    train_imgs, train_labels = make_dataset.read_retina_images(which_set='train')
    test_imgs, test_labels = make_dataset.read_retina_images(which_set='test')
    N_train, m_train, n_train, chan_train = train_imgs.shape
    N_test, m_test, n_test, chan_test = test_imgs.shape
    print('# of training examples: ' + str(N_train) + '\n# of testing examples:  ' + str(N_test))

    # Using sklearn, train a model using Incremental PCA (out-of-core)
    n_components = 8*8
    batch_size = 5*n_components
    # ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    # ipca = joblib.load('/media/dick/Storage1TB/models/ipca_' + str(n_components) + '.pkl')
    # for i in range(240000, N_train, batch_size):
    #     print('-> Progress: ' + str(float(i)*100/N_train) + '%, ' + str(i) + '/' + str(N_train))
    #     ipca.partial_fit(train_imgs.reshape((N_train, m_train*n_train*chan_train))[i:(i+batch_size)])
    #     # Save the model every 10 iterations
    #     if i in range(10*batch_size, N_train, 10*batch_size):
    #         print('Saving model...')
    #         joblib.dump(ipca, '/media/dick/Storage1TB/models/ipca_' + str(n_components) + '.pkl')
    # print('Incremental PCA complete!')
    #
    # joblib.dump(ipca, '/media/dick/Storage1TB/models/ipca_' + str(n_components) + '.pkl')
    # print('IPCA model saved.')

    # Load the trained model
    # ipca = joblib.load('/media/dick/Storage1TB/models/ipca_' + str(n_components) + '.pkl')


    # Create/Open the memory mapped variables
    X_ipca = np.memmap('/media/dick/Storage1TB/transformed/train_ipca.mmap',
                       mode='r', shape=(N_train, n_components), dtype='float')
    X_test_ipca = np.memmap('/media/dick/Storage1TB/transformed/test_ipca.mmap',
                            mode='r', shape=(N_test, n_components), dtype='float')

    # Transform the data using the model
    # N1 = 1580670
    # N2 = 2410920

    # Create in mass
    # X_ipca[:N1] = ipca.transform(train_imgs.reshape((N_train, m_train*n_train))[:N1])
    # X_test_ipca[:N2] = ipca.transform(test_imgs.reshape((N_test, m_test*n_test))[:N2])

    # Create in batch
    # batch = 1000
    # print('Training data:')
    # for i in range(1580000, N_train, batch):
    #     print('\r-> Progress: ' + str((float(i)+1)*100/N_train) + '%, ' + str(i) + '/' + str(N_train)),
    #     temp = N_train-i
    #     if temp < batch:
    #         X_ipca[i:(i+temp)] = ipca.transform(train_imgs[i:(i+temp)].reshape((temp, m_train*n_train*chan_train)))
    #     else:
    #         X_ipca[i:(i+batch)] = ipca.transform(train_imgs[i:(i+batch)].reshape((batch, m_train*n_train*chan_train)))
    # print('Testing data:')
    # for i in range(0, N_test, batch):
    #     print('\r-> Progress: ' + str((float(i)+1)*100/N_test) + '%, ' + str(i) + '/' + str(N_test)),
    #     temp = N_test-i
    #     if temp < batch:
    #         X_test_ipca[i:(i+temp)] = ipca.transform(test_imgs[i:(i+temp)].reshape((temp, m_test*n_test*chan_test)))
    #     else:
    #         X_test_ipca[i:(i+batch)] = ipca.transform(test_imgs[i:(i+batch)].reshape((batch, m_test*n_test*chan_test)))
    #
    # print('Data transformed!')


    # for verification, find the first index of each label
    # idx = [next(x[0] for x in enumerate(train_labels) if x[1] == i) for i in range(5)]
    # for i in range(5):
    #     print(int(train_labels[idx[i]]), X_ipca[idx[i]])
    #     print(ipca.transform(train_imgs.reshape((N_train, m_train*n_train))[idx[i]]))
    #     # X_ipca[idx[i]] = ipca.transform(train_imgs.reshape((N_train, m_train*n_train))[idx[i]])

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
    # joblib.dump(NN, '/media/dick/Storage1TB/models/25NN_64components.pkl')

    # Load the NN model
    NN = joblib.load('/media/dick/Storage1TB/models/25NN_64components.pkl')
    # NN = joblib.load('/media/dick/Storage1TB/models/NN_25.pkl')

    # Test kNN on training data
    # distances, indices = NN.kneighbors(X_ipca[:10])
    # print('Indices: ' + str(indices))
    # print('Distances: ' + str(distances))
    # print(str(train_labels[indices[0]]))

    # Find kNN on train data: num_test should be a multiple of the number of sub-images per image
    # num_train = 45*10
    # target_names = ['0', '1', '2', '3', '4']
    # y1 = []
    # for train in X_ipca[:num_train]:
    #     distances, indices = NN.kneighbors(train)
    #     if np.sum(distances) > 0:
    #         print(distances)
    #         print(indices)
    #         print(train_labels[indices])
    #         raw_input('')
    #     # print(np.max(train_labels[indices][0]), most_common(train_labels[indices][0]))
    #     # y1.append(int(most_common(train_labels[indices[:4]][0])))
    #     y1.append(np.max(train_labels[indices][0]))
    #     # hist, _ = np.histogram(train_labels[indices][0], 5, [0, 4])
    #     # hist = np.divide(hist.astype('float'), np.sum(hist))
    #     # print(hist)
    # print(len(y1))
    # # report = classification_report(train_labels[:num_train], y1, target_names=target_names)
    # # print(report)
    # cm = confusion_matrix(train_labels[:num_train], y1)
    # print(cm)
    #
    # raise SystemExit

    # Find kNN on test data
    black_thres = 0.75
    y_long = []
    i = 0
    for test in X_test_ipca:
        hist_test = np.histogram(test, 5, [0, 4])
        if np.sum(hist_test[1]) > 0:
            hist_test = np.divide(np.asarray(hist_test[1]).astype('float'), sum(hist_test[1]))
            if hist_test[0] < black_thres:
                indices = NN.kneighbors(test, return_distance=False)
                # y.append(int(most_common(train_labels[indices[:4]][0])))
                y_long.append(np.max(train_labels[indices][0]))
            else:
                y_long.append(0)
        i += 1
        print('\r-> Progress: ' + str(i) + '/' + str(N_test)),
    # joblib.dump(indices, '/media/dick/External/KaggleRetina/models/testNNindices_25.pkl')
    y = [most_common(y_long[i:(i+45)]) for i in range(0, len(y_long), 45)]
    y_max = [np.max(y_long[i:(i+45)]) for i in range(0, len(y_long), 45)]
    make_test_preds(y, '072415_mostcommon_NN')
    make_test_preds(y_max, '072415_max_NN')
    raise SystemExit



    # create a CLAHE object (Arguments are optional) to perform non-global contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    img = np.reshape(test_imgs[4], (200, 200))
    plt.subplot(1, 2, 1), plt.imshow(img, 'gray'), plt.axis('off'), plt.title('image')
    plt.subplot(1, 2, 2), plt.imshow(clahe.apply(img), 'gray'), plt.axis('off'), plt.title('image contrast normalized')
    plt.show()
    cv2.waitKey(0)
    raise SystemExit

    print('Number of sub-images to test: ' + str(len(indices)))
    preds = make_predictions(train_labels, indices)
    print('Number of predictions: ' + str(len(preds)))

    print('Unique labels from training images: ' + str(np.unique(train_labels)))
    print('Unique labels test images: ' + str(np.unique(np.unique(preds))))

    # Make test set labels for submission
    with open('./labels/test_labels_IPCA2.csv', 'wb') as f:
        writer = csv.writer(f)
        for i in range(N_test):
            if i == 0:
                writer.writerows([['image', 'level']])
            writer.writerows([[test_file_names[i][:-5], preds[i]]])
            print('\rProgress: ' + str(int(float(i)/N_test))),

    print("\nCSV file containing image-label pairs was created.")


