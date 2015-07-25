import os
import numpy as np
import make_dataset


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

    # Create/Open the memory mapped variables
    X_ipca = np.memmap('/media/dick/Storage1TB/transformed/train_ipca.mmap',
                       mode='r', shape=(N_train, 64), dtype='float')
    X_test_ipca = np.memmap('/media/dick/Storage1TB/transformed/test_ipca.mmap',
                            mode='r', shape=(N_test, 64), dtype='float')

    # For training data (contains labels {1,2,3,4,5})
    print('Training data.')
    with open('/media/dick/Storage1TB/transformed/train_vw2.dat', 'wb') as f:
        for i in range(N_train):
            byte = str(1+int(train_labels[i])) + ' | '
            for j in range(64):
                byte += str(X_ipca[i][j]) + ' '
            byte += '\n'
            f.write(byte)
            print('\r-> Progress: ' + str(i) + '/' + str(N_train)),
    raise SystemExit
    # For testing data (no labels)
    print('\nTesting data.')
    with open('/media/dick/Storage1TB/transformed/test_vw2.dat', 'wb') as f:
        for i in range(N_test):
            byte = ' | '
            for j in range(64):
                byte += str(X_test_ipca[i][j]) + ' '
            byte += '\n'
            f.write(byte)
            print('\r-> Progress: ' + str(i) + '/' + str(N_test)),
