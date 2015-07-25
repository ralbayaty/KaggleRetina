import time
import os
import csv
import numpy as np
import cPickle as pickle

from wabbit_wappa import *
import make_dataset


def make_features(data):
    length = 64
    features = []
    for i in range(length):
        features.append(str(data[i]))
    return features


def make_labels(y_labels, num=45):
    y = np.empty(num*len(y_labels))
    for i in range(len(y_labels)):
        for j in range(num):
            y[j] = y_labels[i]
    return y


if __name__ == '__main__':
    test_file_names = os.listdir('/media/dick/Storage1TB/test')

    train_imgs, train_labels = make_dataset.read_retina_images(which_set='train')
    test_imgs, test_labels = make_dataset.read_retina_images(which_set='test')
    N_train, m_train, n_train, chan_train = train_imgs.shape
    N_test, m_test, n_test, chan_test = test_imgs.shape
    print('# of training examples: ' + str(N_train) + '\n# of testing examples:  ' + str(N_test))
    # of training examples: 1580670
    # of testing examples:  2410920

    # Generate the list of labels since we made 45 patches of each image
    # y_train = make_labels(train_labels)
    # out = open('labels/y_train.pkl', 'wb')
    # pickle.dump(y_train, out)
    # out.close()

    # y_train = pickle.load(open('labels/y_train.pkl', 'rb'))
    y_train = train_labels
    print(len(y_train))

    # Create/Open the memory mapped variables
    X_train = np.memmap('/media/dick/Storage1TB/transformed/train_ipca.mmap',
                        mode='r', shape=(N_train, 64), dtype='float')

    # vw = VW(loss_function='logistic')
    # print("VW command:", vw.command)

    num_train = int(round(0.8*N_train))
    num_valid = N_train - num_train

    # print('Training...')
    # for i in range(num_train):
    #     features = make_features(X_train[i])
    #     # print(features)
    #     # if i < 5000:
    #     #     continue
    #     vw.send_example(int(y_train[i]), features=features)
    #     # time.sleep(0.01)
    #     if i in range(0, num_train, 1000):
    #         vw.save_model('vm_train80.saved.model')
    #         vw.close()
    #         vw = VW(loss_function='logistic', i='vm_train80.saved.model')
    #         print('\nSaved model.')
    #     print('\r-> Progress: ' + str(i) + '/' + str(num_train)),
    # vw.save_model('vm_train80.saved.model')
    # vw.close()

    vw = VW(loss_function='logistic', i='vm_train80.saved.model')
    print('Validating...')
    num_good_tests = 0
    for i in range(num_train, N_train):
        features = make_features(X_train[i])
        # Give the features to the model, witholding the label
        prediction = vw.get_prediction(features).prediction
        print(y_train[i], prediction, int(round(prediction)))
        print('\r-> Progress: ' + str(i-num_train) + '/' + str(num_valid)),
        if y_train[i] - int(round(prediction)) == 0:
            num_good_tests += 1
    print("Correctly predicted", num_good_tests, "out of", num_valid)
    # print(quadratic_weighted_kappa(np.ravel(train_labels[:num_train]), classifier.predict(X_ipca[:num_train])))

    print("Train on the rest of the validation examples.")
    for i in range(num_train, N_train):
        features = make_features(X_train[i])
        vw.send_example(y_train[i], features=features)
        if i in range(num_train, N_train, 1000):
            vw.save_model('vm_train100.saved.model')
            vw.close()
            vw = VW(loss_function='logistic', i='vm_train100.saved.model')
            print('\nSaved model.')
    vw.save_model('vm_train100.saved.model')

    X_test = np.memmap('/media/dick/Storage1TB/transformed/test_ipca.mmap',
                       mode='r', shape=(N_test, 64), dtype='float')

    # preds = np.empty(N_test)
    # for i in range(num_valid):
    #     features = make_features(X_test[i])
    #     preds[i] = vw.get_prediction(features).prediction

    # Make test set labels for submission
    with open('./labels/test_labels_vm_log.csv', 'wb') as f:
        writer = csv.writer(f)
        for i in range(N_test):
            if i == 0:
                writer.writerows([['image', 'level']])
            preds = vw.get_prediction(make_features(X_test[i])).prediction
            writer.writerows([[test_file_names[i][:-5], int(round(preds))]])
            print('\rProgress: ' + str(int(float(i)/N_test))),

    print("\nCSV file containing image-label pairs was created.")

    vw.close()
