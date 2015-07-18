import os
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


if __name__ == '__main__':

    test_file_names = os.listdir('/media/dick/Storage1TB/test')
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
    n_components = 8*8
    batch_size = 5*n_components
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for i in range(0, N_train, batch_size):
        print('-> Progress: ' + str(float(i)*100/N_train))
        ipca.partial_fit(train_imgs.reshape((N_train, m_train*n_train))[i:(i+batch_size)])

    # ipca.fit(train_imgs.reshape((N_train, m_train*n_train))[:50000])
    print('Incremental PCA complete!')

    joblib.dump(ipca, '/media/dick/External/KaggleRetina/models/ipca_' + str(n_components) + '.pkl')
    print('IPCA model saved.')

    # Create/Open the memory mapped variables
    X_ipca = np.memmap('/media/dick/External/KaggleRetina/train_ipca.mmap',
               mode='w+', shape=(N_train, n_components))
    X_test_ipca = np.memmap('/media/dick/External/KaggleRetina/test_ipca.mmap',
               mode='w+', shape=(N_test, n_components))


    # Load the trained model
    # ipca = joblib.load('/media/dick/External/KaggleRetina/models/ipca.pkl')

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

    # Find kNN on train data: num_test should be a multiple of the number of sub-images per image
    num_train = 45*100
    target_names = ['0', '1', '2', '3', '4']
    y1 = []
    for train in X_ipca[:num_train]:
        distances, indices = NN.kneighbors(train)
        if np.sum(distances) > 0:
            print(distances)
            print(indices)
            print(train_labels[indices])
            raw_input('')
        # print(np.max(train_labels[indices][0]), most_common(train_labels[indices][0]))
        y1.append(int(most_common(train_labels[indices[:4]][0])))
        # hist, _ = np.histogram(train_labels[indices][0], 5, [0, 4])
        # hist = np.divide(hist.astype('float'), np.sum(hist))
        # print(hist)
    report = classification_report(train_labels[:num_train], y1, target_names=target_names)
    cm = confusion_matrix(train_labels[:num_train], y1)
    print(report)
    print(cm)

    raise SystemExit

    # Find kNN on test data: num_test should be a multiple of the number of sub-images per image
    num_test = 45*10
    black_thres = 0.75
    y = []
    for test in X_test_ipca[:45]:
        hist_test = np.histogram(test, 5, [0, 4])
        if sum(hist_test) > 0:
            hist_test = np.divide(hist_test.astype('float'), sum(hist_test))
            if hist_test[0] < black_thres:
                indices = NN.kneighbors(test, return_distance=False)
                y.append(int(most_common(train_labels[indices[:4]][0])))
            else:
                y.append(-1)


    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    img = np.reshape(test_imgs[4], (200, 200))
    plt.subplot(1, 2, 1), plt.imshow(img, 'gray'), plt.axis('off'), plt.title('image')
    plt.subplot(1, 2, 2), plt.imshow(clahe.apply(img), 'gray'), plt.axis('off'), plt.title('image contrast normalized')
    plt.show()
    cv2.waitKey(0)
    raise SystemExit

    # joblib.dump(indices, '/media/dick/External/KaggleRetina/models/testNNindices_25.pkl')

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


