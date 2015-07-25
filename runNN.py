import os
from sklearn.externals import joblib
import time
import numpy as np
import csv
import cv2
from collections import Counter
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import make_dataset
from kappa import *


def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def make_predictions(labels, idx, n=45):
    preds = [most_common(labels[idx[x]]) for x in range(len(idx))]
    predictions = [most_common(list(preds[x:(x+n)])) for x in range(0, len(idx), n)]
    return predictions


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

    train_file_names = os.listdir('/media/dick/Storage1TB/train')
    N_train = len(train_file_names)
    print(str(N_test) + ' test files.\n' + str(N_train) + ' train files.')

    train_imgs, train_labels = make_dataset.read_retina_images(which_set='train')
    test_imgs, test_labels = make_dataset.read_retina_images(which_set='test')
    N_train, m_train, n_train, chan_train = train_imgs.shape
    N_test, m_test, n_test, chan_test = test_imgs.shape
    print('# of testing examples:  ' + str(N_test) + '\n# of training examples: ' + str(N_train))

    bins = [[], [], [], [], []]
    for x in enumerate(train_labels):
        bins[x[1]].append(x[0])
    occurences = [len(x) for x in bins]
    print(occurences)

    # tot = 1000
    # y = np.empty(5*tot)
    # y[0:tot].fill(0)
    # y[tot:2*tot].fill(1)
    # y[2*tot:3*tot].fill(2)
    # y[3*tot:4*tot].fill(3)
    # y[4*tot:5*tot].fill(4)
    #
    # for i in range(0, 5):
    #     if i == 0:
    #         X_train = np.reshape(train_imgs[bins[i][:tot]], (tot, m_train*n_train*chan_train))
    #     else:
    #         X_train = np.append(X_train, np.reshape(train_imgs[bins[i][:tot]], (tot, m_train*n_train*chan_train)), axis=0)


    # Create/Open the memory mapped variables
    n_components = 8*8
    X_ipca = np.memmap('/media/dick/Storage1TB/transformed/train_ipca.mmap',
                       mode='r', shape=(N_train, n_components), dtype='float')
    X_test_ipca = np.memmap('/media/dick/Storage1TB/transformed/test_ipca.mmap',
                            mode='r', shape=(N_test, n_components), dtype='float')
    #
    # print('Shape of training images after IPCA: ' + str(X_ipca.shape))
    # print('Shape of testing images after IPCA: ' + str(X_test_ipca.shape))

    # idx = [next(x[0] for x in enumerate(train_labels) if x[1] == i) for i in range(5)]
    # for i in range(5):
    #     print(int(train_labels[idx[i]]), X_ipca[idx[i]])
    #     plt.subplot(1, 5, i+1), plt.imshow(np.reshape(X_ipca[idx[i]], (8, 8)), 'gray'), plt.axis('off'), plt.title(i)
    # plt.ion()
    # plt.show()

    # print(np.sum(X_ipca[idx[0]]-X_ipca[idx[1]]))
    # print(np.sum(X_ipca[idx[1]]-X_ipca[idx[2]]))
    # print(train_imgs[0].shape)
    # for i in range(0, 100, 10):
    #     plt.imshow(train_imgs[i]), plt.axis('off'), plt.title(i)
    #     plt.draw()
    #     time.sleep(0.05)
    #
    # print(test_imgs[0].shape)
    # for i in range(0, 100, 10):
    #     plt.imshow(test_imgs[i]), plt.axis('off'), plt.title(i)
    #     plt.draw()
    #     time.sleep(0.05)
    #
    # raise SystemExit

    # Models we will use
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)

    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    ###############################################################################
    # Training

    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    rbm.learning_rate = 0.6
    rbm.n_iter = 1000
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = n_components
    logistic.C = 100.0

    # Training RBM-Logistic Pipeline
    num_train = N_train
    # X_ipca = np.reshape(train_imgs[:num_train], (num_train,  m_train*n_train*chan_train))

    # classifier.fit(X_ipca[:num_train], np.ravel(train_labels[:num_train]))

    # Training Logistic regression
    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(X_ipca[:num_train], np.ravel(train_labels[:num_train]))
    ###############################################################################
    # Evaluation

    # print("Logistic regression using RBM features:\n%s\n" % (
    #     metrics.classification_report(
    #         np.ravel(train_labels[:num_train]),
    #         classifier.predict(X_ipca[:num_train]))))
    #
    # print("Logistic regression using raw pixel features:\n%s\n" % (
    #     metrics.classification_report(
    #         np.ravel(train_labels[:num_train]),
    #         logistic_classifier.predict(X_ipca[:num_train]))))
    #
    # print("Logistic regression using RBM features:\n%s\n" % (
    #     metrics.confusion_matrix(
    #         np.ravel(train_labels[:num_train]),
    #         classifier.predict(X_ipca[:num_train]))))
    #
    # print("Logistic regression using raw pixel features:\n%s\n" % (
    #     metrics.confusion_matrix(
    #         np.ravel(train_labels[:num_train]),
    #         logistic_classifier.predict(X_ipca[:num_train]))))

    # print(quadratic_weighted_kappa(np.ravel(train_labels[:num_train]), classifier.predict(X_ipca[:num_train])))
    # print(quadratic_weighted_kappa(np.ravel(train_labels[:num_train]), logistic_classifier.predict(X_ipca[:num_train])))

    end = N_test
    y_long = np.empty(end)

    for j in range(0, end, 1000):
        temp = end-j
        if temp < 1000:
            # y_long[j:(j+temp)] = classifier.predict(np.reshape(X_test_ipca[j:(j+temp)], (temp, m_test*n_test*chan_test)))
            y_long[j:(j+temp)] = logistic_classifier.predict(X_test_ipca[j:(j+temp)])
        else:
            # y_long[j:(j+1000)] = classifier.predict(np.reshape(X_test_ipca[j:(j+1000)], (1000, m_test*n_test*chan_test)))
            y_long[j:(j+1000)] = logistic_classifier.predict(X_test_ipca[j:(j+1000)])
        print(float(j)*100/end)
    y = [most_common(y_long[i:(i+45)]) for i in range(0, end, 45)]
    y_max = [np.max(y_long[i:(i+45)]) for i in range(0, end, 45)]
    joblib.dump(y_long, '/media/dick/Storage1TB/labels/y_long_072415.pkl')
    make_test_preds(y, '072415_mostcommon_log')
    make_test_preds(y_max, '072415_max_log')

    raise SystemExit

    # Find kNN on train data: num_test should be a multiple of the number of sub-images per image
    num_train = 45*10
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
        # y1.append(int(most_common(train_labels[indices[:4]][0])))
        y1.append(np.max(train_labels[indices][0]))
        # hist, _ = np.histogram(train_labels[indices][0], 5, [0, 4])
        # hist = np.divide(hist.astype('float'), np.sum(hist))
        # print(hist)
    print(len(y1))
    # report = classification_report(train_labels[:num_train], y1, target_names=target_names)
    # print(report)
    cm = confusion_matrix(train_labels[:num_train], y1)
    print(cm)

    raise SystemExit

    # Find kNN on test data: num_test should be a multiple of the number of sub-images per image
    num_test = 45*1
    black_thres = 0.75
    y = []
    for test in X_test_ipca[:45]:
        hist_test = np.histogram(test, 5, [0, 4])
        if sum(hist_test) > 0:
            hist_test = np.divide(hist_test.astype('float'), sum(hist_test))
            if hist_test[0] < black_thres:
                indices = NN.kneighbors(test, return_distance=False)
                # y.append(int(most_common(train_labels[indices[:4]][0])))
                y.append(np.max(train_labels[indices][0]))
            else:
                y.append(-1)
    # joblib.dump(indices, '/media/dick/External/KaggleRetina/models/testNNindices_25.pkl')

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


