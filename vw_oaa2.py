import time
import os
import csv
import numpy as np
import cPickle as pickle
from kappa import *

from wabbit_wappa import *
import make_dataset


def make_features(data):
    length = 64
    features = []
    for i in range(length):
        features.append(str(data[i]))
    return features

# vw -d train_vw.data --oaa 5 -f oaa.model -b 30 --passes 100
if __name__ == '__main__':
    test_file_names = os.listdir('/media/dick/Storage1TB/test')

    train_imgs, train_labels = make_dataset.read_retina_images(which_set='train')
    test_imgs, test_labels = make_dataset.read_retina_images(which_set='test')
    N_train, m_train, n_train, chan_train = train_imgs.shape
    N_test, m_test, n_test, chan_test = test_imgs.shape
    print('# of training examples: ' + str(N_train) + '\n# of testing examples:  ' + str(N_test))
    # of training examples: 1580670
    # of testing examples:  2410920

    y_train = train_labels
    print(len(y_train))

    # Create/Open the memory mapped variables
    X_train = np.memmap('/media/dick/Storage1TB/transformed/train_ipca.mmap',
                        mode='r', shape=(N_train, 64), dtype='float')

    num_train = int(round(0.8*N_train))
    num_valid = N_train - num_train

    # print('Training...')
    vw = VW(oaa=5)
    for i in range(num_train):
        features = make_features(X_train[i])
        label = int(y_train[i])+1
        vw.send_example(label, features=features)
        # if i in range(100, num_train, 1000):
        #     vw.save_model('vm_train80.saved.model')
        #     vw.close()
        #     vw = VW(oaa=5, i='vm_train80.saved.model')
        #     print('\nSaved model.')
        print('\r-> Progress: ' + str(i) + '/' + str(num_train)),
    vw.save_model('vm_train80.saved.model')
    vw.close()

    print('Validating...')
    num_good_tests = 0
    output = np.array((0, 0, 0, 0, 0))
    preds = []
    for i in range(num_train, N_train):
        features = make_features(X_train[i])
        vw = VW(oaa=5, i='vm_train80.saved.model')
        prediction = vw.get_prediction(features).prediction
        vw.close()
        preds.append(prediction)
        print(int(y_train[i]), prediction)
        if prediction == None:
            continue
        print('\r-> Progress: ' + str(i-num_train) + '/' + str(num_valid)),
        if int(y_train[i])+1 - prediction == 0:
            num_good_tests += 1
    print("Correctly predicted", num_good_tests, "out of", num_valid)

    # Determine the weighted kappa score for the validation set (1 is the best)
    print(quadratic_weighted_kappa(np.ravel(train_labels[num_train:N_train]), preds))

    print("Train on the rest of the validation examples.")
    vw = VW(oaa=5, i='vm_train80.saved.model')
    for i in range(num_train, N_train):
        features = make_features(X_train[i])
        label = int(y_train[i])+1
        vw.send_example(label, features=features)
        if i in range(num_train, N_train, 1000):
            vw.save_model('vm_train100.saved.model')
            vw.close()
            vw = VW(oaa=5, i='vm_train100.saved.model')
            print('\nSaved model.')
        print('\r-> Progress: ' + str(i) + '/' + str(num_train)),
    vw.save_model('vm_train100.saved.model')
    vw.close()

    X_test = np.memmap('/media/dick/Storage1TB/transformed/test_ipca.mmap',
                       mode='r', shape=(N_test, 64), dtype='float')

    # preds = np.empty(N_test)
    # for i in range(num_valid):
    #     features = make_features(X_test[i])
    #     preds[i] = vw.get_prediction(features).prediction

    output = np.array((0, 0, 0, 0, 0))
    # Make test set labels for submission
    with open('./labels/test_labels_vm_oaa_07242015.csv', 'wb') as f:
        writer = csv.writer(f)
        for i in range(N_test):
            if i == 0:
                writer.writerows([['image', 'level']])
            features = make_features(X_test[i])
            preds = vw.get_prediction(features).prediction
            writer.writerows([[test_file_names[i][:-5], int(round(preds))]])
            print('Progress: ' + str(int(float(i)/N_test)))

    print("\nCSV file containing image-label pairs was created.")
