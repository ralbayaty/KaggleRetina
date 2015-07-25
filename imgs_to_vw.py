import time
import numpy as np

from wabbit_wappa import *
import make_dataset


def make_features(data):
    length = 64
    features = []
    for i in range(length):
        features.append(str(data[i]))
    return features


if __name__ == '__main__':

    train_imgs, train_labels = make_dataset.read_retina_images(which_set='train')
    test_imgs, test_labels = make_dataset.read_retina_images(which_set='test')
    N_train, m_train, n_train, chan_train = train_imgs.shape
    N_test, m_test, n_test, chan_test = test_imgs.shape
    print('# of training examples: ' + str(N_train) + '\n# of testing examples:  ' + str(N_test))

    # Create/Open the memory mapped variables
    X_ipca = np.memmap('/media/dick/Storage1TB/transformed/train_ipca.mmap',
                       mode='r', shape=(N_train, 64), dtype='float')
    # X_test_ipca = np.memmap('/media/dick/Storage1TB/transformed/test_ipca.mmap',
    #                         mode='r', shape=(N_test, n_components), dtype='float')

    y_train = train_labels

    print("Start a Vowpal Wabbit learner in logistic regression mode")
    vw = VW(oaa=5)
    # ./vw -d train.txt -f lg.vw --loss_function logistic
    # Print the command line used for the VW process
    print("VW command:", vw.command)

    print("Now generate 10 training examples, feeding them to VW one by one.")
    for i in range(10):
        features = make_features(X_ipca[i])
        vw.send_example(int(y_train[i])+1, features=features)

    print("How well trained is our model?  Let's make 100 tests.")
    num_tests = 100
    num_good_tests = 0
    for i in range(num_tests):
        features = make_features(X_ipca[i])
        # Give the features to the model, witholding the label
        prediction = vw.get_prediction(features).prediction
        # Test whether the floating-point prediction is in the right direction (signs agree)
        print(int(y_train[i]), prediction)
        if y_train[i] - int(round(prediction)) == 0:
            num_good_tests += 1
    print("Correctly predicted", num_good_tests, "out of", num_tests)

    print("We can go on training, without restarting the process.  Let's train on 1,000 more examples.")
    for i in range(1000):
        features = make_features(X_ipca[i])
        vw.send_example(y_train[i], features=features)

    print("Now how good are our predictions?")
    num_tests = 100
    num_good_tests = 0
    for i in range(num_tests):
        features = make_features(X_ipca[i])
        # Give the features to the model, witholding the label
        prediction = vw.get_prediction(features).prediction
        # Test whether the floating-point prediction is in the right direction
        if y_train[i] - int(round(prediction)) == 0:
            num_good_tests += 1
    print("Correctly predicted", num_good_tests, "out of", num_tests)

    filename = 'test.saved.model'
    print("We can save the model at any point in the process.")
    print("Saving now to", filename)
    vw.save_model(filename)
    vw.close()

    print("We can reload our model using the 'i' argument:")
    vw2 = VW(loss_function='logistic', i=filename)
    print("""vw2 = VW(loss_function='logistic', i=filename)""")
    print("VW command:", vw2.command)

    print("How fast can we train and test?")
    num_examples = 1000
    # Generate examples ahead of time so we don't measure that overhead
    examples = [(y_train[i], make_features(X_ipca[i])) for i in range(num_examples)]
    print("Training on", num_examples, "examples...")
    start_time = time.time()
    for example in examples:
        label, features = example
        # Turning off parse_result mode speeds up training when we
        # don't care about the result of each example
        vw2.send_example(label, features=features, parse_result=False)
    duration = time.time() - start_time
    frequency = num_examples / duration
    print("Trained", frequency, "examples per second")

    start_time = time.time()
    print("Testing on", num_examples, "examples...")
    for example in examples:
        label, features = example
        # Give the features to the model, witholding the label
        prediction = vw2.get_prediction(features).prediction
    duration = time.time() - start_time
    frequency = num_examples / duration
    print("Tested", frequency, "examples per second")

    # # For training data (contains labels)
    # with open('/media/dick/Storage1TB/transformed/train.vw', 'wb') as f:
    #     length = 64
    #     byte = str(int(train_labels[0])) + ' | '
    #     for i in range(length):
    #         byte += str(X_ipca[0][i]) + ' '
    #
    #     print(byte)
    #     f.write(byte)
    #
    # # For testing data (no labels)
    # with open('/media/dick/Storage1TB/transformed/test.vw', 'wb') as f:
    #     length = 64
    #     byte = ' | '
    #     for i in range(length):
    #         byte += str(X_test_ipca[0][i]) + ' '
    #
    #     print(byte)
    #     f.write(byte)