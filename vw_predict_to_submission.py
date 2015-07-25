import os
import time
from sklearn.externals import joblib
import numpy as np
import csv
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

import make_dataset
from kappa import *


def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


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

    train_imgs, train_labels = make_dataset.read_retina_images(which_set='train')
    test_imgs, test_labels = make_dataset.read_retina_images(which_set='test')
    N_train, m_train, n_train, chan_train = train_imgs.shape
    N_test, m_test, n_test, chan_test = test_imgs.shape
    print('# of training examples: ' + str(N_train) + '\n# of testing examples:  ' + str(N_test))

    train_labels = [most_common(train_labels[i:(i+45)]) for i in range(0, len(train_labels), 45)]
    bins = [[], [], [], [], []]
    for x in enumerate(train_labels):
        bins[x[1]].append(x[0])
    occurences = [len(x) for x in bins]
    print(occurences)

    y_long = []
    with open('/media/dick/Storage1TB/transformed/oaa.predict', 'rb') as predictions:
        for line in predictions:
            y_long.append(int(float(line.strip()))-1)
        y_common = [most_common(y_long[i:(i+45)]) for i in range(0, len(y_long), 45)]
        y_max = [np.max(y_long[i:(i+45)]) for i in range(0, len(y_long), 45)]
        print(len(y_common), len(y_max))
        print(np.min(y_common), np.min(y_max))
        print(np.max(y_common), np.max(y_max))

        bins = [[], [], [], [], []]
        for x in enumerate(y_common):
            bins[x[1]].append(x[0])
        occurences = [len(x) for x in bins]
        print(occurences)
        bins = [[], [], [], [], []]
        for x in enumerate(y_max):
            bins[x[1]].append(x[0])
        occurences = [len(x) for x in bins]
        print(occurences)

        # print(quadratic_weighted_kappa(np.ravel(train_labels), y_common))
        # print(quadratic_weighted_kappa(np.ravel(train_labels), y_max))
        # print("y_common:\n%s\n" % (classification_report(np.ravel(train_labels), y_common)))
        # print("y_max:\n%s\n" % (classification_report(np.ravel(train_labels), y_max)))

        make_test_preds(y_common, '072515_mostcommon_oaa2')
        make_test_preds(y_max, '072515_max_oaa2')

