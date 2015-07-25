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

    print('Training...')
    # # Label 0
    # vw0 = VW(loss_function='logistic')
    # for i in range(num_train):
    #     features = make_features(X_train[i])
    #     label = int(y_train[i])
    #     if label == 0:
    #         vw0.send_example(1, features=features)
    #     else:
    #         vw0.send_example(-1, features=features)
    #
    #     if i in range(0, num_train, 1000):
    #         vw0.save_model('vm0_train80.saved.model')
    #         vw0.close()
    #         vw0 = VW(loss_function='logistic', i='vm0_train80.saved.model')
    #         print('\nSaved model.')
    #     print('\r-> Progress: ' + str(i) + '/' + str(num_train)),
    # vw0.save_model('vm0_train80.saved.model')
    # vw0.close()
    #
    # # Label 1
    # vw1 = VW(loss_function='logistic')
    # for i in range(num_train):
    #     features = make_features(X_train[i])
    #     label = int(y_train[i])
    #     if label == 1:
    #         vw1.send_example(1, features=features)
    #     else:
    #         vw1.send_example(-1, features=features)
    #
    #     if i in range(0, num_train, 1000):
    #         vw1.save_model('vm1_train80.saved.model')
    #         vw1.close()
    #         vw1 = VW(loss_function='logistic', i='vm1_train80.saved.model')
    #         print('\nSaved model.')
    #     print('\r-> Progress: ' + str(i) + '/' + str(num_train)),
    # vw1.save_model('vm1_train80.saved.model')
    # vw1.close()
    #
    # # Label 2
    # vw2 = VW(loss_function='logistic')
    # for i in range(num_train):
    #     features = make_features(X_train[i])
    #     label = int(y_train[i])
    #     if label == 2:
    #         vw2.send_example(1, features=features)
    #     else:
    #         vw2.send_example(-1, features=features)
    #
    #     if i in range(0, num_train, 1000):
    #         vw2.save_model('vm2_train80.saved.model')
    #         vw2.close()
    #         vw2 = VW(loss_function='logistic', i='vm2_train80.saved.model')
    #         print('\nSaved model.')
    #     print('\r-> Progress: ' + str(i) + '/' + str(num_train)),
    # vw2.save_model('vm2_train80.saved.model')
    # vw2.close()

    # Label 3
    vw3 = VW(loss_function='logistic')
    for i in range(num_train):
        features = make_features(X_train[i])
        label = int(y_train[i])
        if label == 3:
            vw3.send_example(1, features=features)
        else:
            vw3.send_example(-1, features=features)

        if i in range(0, num_train, 1000):
            vw3.save_model('vm3_train80.saved.model')
            vw3.close()
            vw3 = VW(loss_function='logistic', i='vm3_train80.saved.model')
            print('\nSaved model.')
        print('\r-> Progress: ' + str(i) + '/' + str(num_train)),
    vw3.save_model('vm3_train80.saved.model')
    vw3.close()

    # Label 4
    vw4 = VW(loss_function='logistic')
    for i in range(num_train):
        features = make_features(X_train[i])
        label = int(y_train[i])
        if label == 4:
            vw4.send_example(1, features=features)
        else:
            vw4.send_example(-1, features=features)

        if i in range(0, num_train, 1000):
            vw4.save_model('vm4_train80.saved.model')
            vw4.close()
            vw4 = VW(loss_function='logistic', i='vm4_train80.saved.model')
            print('\nSaved model.')
        print('\r-> Progress: ' + str(i) + '/' + str(num_train)),
    vw4.save_model('vm4_train80.saved.model')
    vw4.close()

    print('Validating...')
    num_good_tests = 0
    output = np.array((0, 0, 0, 0, 0))
    for i in range(num_train, N_train):
        features = make_features(X_train[i])

        vw0 = VW(loss_function='logistic', i='vm0_train80.saved.model')
        output[0] = vw0.get_prediction(features).prediction
        vw0.close()

        vw1 = VW(loss_function='logistic', i='vm1_train80.saved.model')
        output[1] = vw1.get_prediction(features).prediction
        vw1.close()

        vw2 = VW(loss_function='logistic', i='vm2_train80.saved.model')
        output[2] = vw2.get_prediction(features).prediction
        vw2.close()

        vw3 = VW(loss_function='logistic', i='vm3_train80.saved.model')
        output[3] = vw3.get_prediction(features).prediction
        vw3.close()

        vw4 = VW(loss_function='logistic', i='vm4_train80.saved.model')
        output[4] = vw4.get_prediction(features).prediction
        vw4.close()

        print(np.where(output >= 0)[0])
        # Use the last index that was flagged as correct label (highest value)
        try:
            prediction = np.where(output > 0)[0][-1]
        except IndexError:
            prediction = 0

        print(int(y_train[i]), prediction)
        print('\r-> Progress: ' + str(i-num_train) + '/' + str(num_valid)),
        if int(y_train[i]) - int(round(prediction)) == 0:
            num_good_tests += 1
    print("Correctly predicted", num_good_tests, "out of", num_valid)
    # print(quadratic_weighted_kappa(np.ravel(train_labels[:num_train]), classifier.predict(X_ipca[:num_train])))

    print("Train on the rest of the validation examples.")
    # Label 0
    vw0 = VW(loss_function='logistic', i='vm0_train80.saved.model')
    for i in range(num_train, N_train):
        features = make_features(X_train[i])
        label = int(y_train[i])
        if label == 0:
            vw0.send_example(1, features=features)
        else:
            vw0.send_example(-1, features=features)

        if i in range(num_train, N_train, 1000):
            vw0.save_model('vm0_train100.saved.model')
            vw0.close()
            vw0 = VW(loss_function='logistic', i='vm0_train100.saved.model')
            print('\nSaved model.')
        print('\r-> Progress: ' + str(i) + '/' + str(num_train)),
    vw0.save_model('vm0_train100.saved.model')
    vw0.close()

    # Label 1
    vw1 = VW(loss_function='logistic', i='vm1_train80.saved.model')
    for i in range(num_train, N_train):
        features = make_features(X_train[i])
        label = int(y_train[i])
        if label == 1:
            vw1.send_example(1, features=features)
        else:
            vw1.send_example(-1, features=features)

        if i in range(num_train, N_train, 1000):
            vw1.save_model('vm1_train100.saved.model')
            vw1.close()
            vw1 = VW(loss_function='logistic', i='vm1_train100.saved.model')
            print('\nSaved model.')
        print('\r-> Progress: ' + str(i) + '/' + str(num_train)),
    vw1.save_model('vm1_train100.saved.model')
    vw1.close()

    # Label 2
    vw2 = VW(loss_function='logistic', i='vm2_train80.saved.model')
    for i in range(num_train, N_train):
        features = make_features(X_train[i])
        label = int(y_train[i])
        if label == 2:
            vw2.send_example(1, features=features)
        else:
            vw2.send_example(-1, features=features)

        if i in range(num_train, N_train, 1000):
            vw2.save_model('vm2_train100.saved.model')
            vw2.close()
            vw2 = VW(loss_function='logistic', i='vm2_train100.saved.model')
            print('\nSaved model.')
        print('\r-> Progress: ' + str(i) + '/' + str(num_train)),
    vw2.save_model('vm2_train100.saved.model')
    vw2.close()

    # Label 3
    vw3 = VW(loss_function='logistic', i='vm3_train80.saved.model')
    for i in range(num_train, N_train):
        features = make_features(X_train[i])
        label = int(y_train[i])
        if label == 3:
            vw3.send_example(1, features=features)
        else:
            vw3.send_example(-1, features=features)

        if i in range(num_train, N_train, 1000):
            vw3.save_model('vm3_train100.saved.model')
            vw3.close()
            vw3 = VW(loss_function='logistic', i='vm3_train100.saved.model')
            print('\nSaved model.')
        print('\r-> Progress: ' + str(i) + '/' + str(num_train)),
    vw3.save_model('vm3_train100.saved.model')
    vw3.close()

    # Label 4
    vw4 = VW(loss_function='logistic', i='vm4_train80.saved.model')
    for i in range(num_train, N_train):
        features = make_features(X_train[i])
        label = int(y_train[i])
        if label == 4:
            vw4.send_example(1, features=features)
        else:
            vw4.send_example(-1, features=features)

        if i in range(num_train, N_train, 1000):
            vw4.save_model('vm4_train100.saved.model')
            vw4.close()
            vw4 = VW(loss_function='logistic', i='vm4_train100.saved.model')
            print('\nSaved model.')
        print('\r-> Progress: ' + str(i) + '/' + str(num_train)),
    vw4.save_model('vm4_train100.saved.model')
    vw4.close()

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
            output[0] = vw0.get_prediction(features).prediction
            output[1] = vw1.get_prediction(features).prediction
            output[2] = vw2.get_prediction(features).prediction
            output[3] = vw3.get_prediction(features).prediction
            output[4] = vw4.get_prediction(features).prediction
            print(np.where(output >= 0)[0])
            # Use the last index that was flagged as correct label (highest value)
            try:
                preds = np.where(output > 0)[0][-1]
            except IndexError:
                preds = 0
            writer.writerows([[test_file_names[i][:-5], int(round(preds))]])
            print('Progress: ' + str(int(float(i)/N_test)))

    print("\nCSV file containing image-label pairs was created.")
