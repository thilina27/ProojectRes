import os
import random
import scipy
import cv2
import numpy as np
import lasagne
from theano import tensor as T
import theano
import time

__author__ = 'thilina'


def readData():
    # parent image directory
    imagedir = '/home/thilina/PycharmProjects/TestSin/Data'
    # probability to insert a data into training or test data set
    p_train = 0.5
    # load all folders in the parent directory
    onlyFolders = [f for f in os.listdir(imagedir) if os.path.isdir(os.path.join(imagedir, f))]

    # loop through each folder
    for i in reversed(range(len(onlyFolders))):
        # get all files in folder
        onlyFiles = [f for f in os.listdir(imagedir + '/' + onlyFolders[i]) if
                     os.path.isfile(os.path.join(imagedir + '/' + onlyFolders[i], f))]
        print(str(len(onlyFiles)) + " images of " + onlyFolders[i])
        count = 1
        # for each file in the folder
        for ii in range(len(onlyFiles)):
            # load image using cv
            Imgtmp = cv2.imread(imagedir + '/' + onlyFolders[i] + '/' + onlyFiles[ii])

            # print (Imgtmp.shape) - (80, 80, 3) if image is color
            # print (Imgtmp[1, 2]) - [230 118 118] array
            # print (Imgtmp[1, 2, 0]) - 230 single element

            # if image is color change it to gray scale
            if Imgtmp.shape[2] == 3:
                Imgtmp = cv2.cvtColor(Imgtmp, cv2.COLOR_BGR2GRAY)

            # apply threshold
            # Imgtmp = cv2.adaptiveThreshold(Imgtmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            # ret, Imgtmp = cv2.threshold(Imgtmp, 127, 255, cv2.THRESH_TRUNC)
            # ret3, Imgtmp = cv2.threshold(Imgtmp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # print (Imgtmp.shape) - (80, 80)
            # resize image to 50 x 50
            Imgtmp = cv2.resize(Imgtmp, (100, 100))

            if count == 1:
                scipy.misc.imsave('/home/thilina/Data/outfile.jpg', Imgtmp)
                count = 2
            # print (Imgtmp[1, 2]) - 140 single value no 3rd dimension [1, 2, 0] will give a error
            # add a dimension to the array to represent the color (1) for the gray scale
            # after changing the color and shape its has form (50,50)
            # after expand its (50,50,1)
            Imgtmpresize = np.expand_dims(Imgtmp, axis=3)
            # print (Imgtmpresize.shape) - (50, 50, 1) added a new dimension
            # print (Imgtmpresize[1, 2, 0]) - 140 single value
            # print (Imgtmpresize[1, 2]) - [140] array since now it has another dimension

            # generate random number to make a test and training data set randomly
            rs = random.random()
            # if its less that p train add image to train data set
            if rs < p_train:
                # if not X_train variable not in the local variables create it using the image file
                if not 'X_train' in locals():
                    # [None, ...] add another dimension to append all images to same array
                    X_train = Imgtmpresize[None, ...]
                    # print (X_train.shape) - (1, 50, 50, 1)
                else:
                    # append the next image to the training data set from axis 0
                    X_train = np.concatenate((X_train, Imgtmpresize[None, ...]), axis=0)

                # create target dat set
                if not 'targets_train' in locals():
                    targets_train = np.array([i])
                    # print (targets_train) - [1] -- array of array
                    # print (targets_train.shape) - (1,)
                else:
                    # append the next target to the same array in the same dimension
                    targets_train = np.concatenate((targets_train, np.array([i])))

            # do same for all test data set
            else:
                if not 'X_test' in locals():
                    X_test = Imgtmpresize[None, ...]
                else:
                    X_test = np.concatenate((X_test, Imgtmpresize[None, ...]), axis=0)
                if not 'targets_test' in locals():
                    targets_test = np.array([i])
                else:
                    targets_test = np.concatenate((targets_test, np.array([i])))

    # typecast targets
    Y_test = targets_test.astype(np.int32)
    Y_train = targets_train.astype(np.int32)

    # Change data type so to use in gpu
    X_test = X_test.astype(np.float32)
    X_train = X_train.astype(np.float32)

    # this is necessary else give a high loss
    X_test -= X_test.mean()
    X_test /= X_test.std()

    X_train -= X_train.mean()
    X_train /= X_train.std()

    # scipy.misc.imsave('/home/thilina/Data/outfile1.jpg', X_train[0, ...])
    print (X_train[0, ...])

    # change dimension in to the desire way (2000, 1, 100, 100)
    # (number of examples , color dimension , height , width )
    X_test = np.transpose(X_test, (0, 3, 1, 2))
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    # todo add a validation data set and give variables to set values that have used

    # return data
    return X_train, Y_train, X_test, Y_test


def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 100, 100),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
        network,
        num_units=40,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        network,
        num_units=3,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def main(num_epochs=10):
    print("Loading data...")

    # read data
    X_train, Y_train, X_test, Y_test = readData()

    print (Y_test)

    # Y_train = np.concatenate((Y_train, Y_test))
    print (Y_train.shape)
    # X_train = np.concatenate((X_train, X_test), axis=0)
    print (X_train.shape)
    # Prepare Theano variables for inputs and targets
    # theano tensor vectors

    # 4d tensor to hold inputs
    input_var = T.tensor4('inputs')

    # tensor vector to hold targets
    target_var = T.ivector('targets')

    #  build the network using tensor vector
    network = build_cnn(input_var)

    # get prediction from the network
    prediction = lasagne.layers.get_output(network)

    # calculate loss using the prediction get from the network and the known targets
    # this is cross entropy
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    # we need only single value to update variables
    loss = loss.mean()

    # Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum
    # to update the weights
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # compile to predict
    predict = theano.function([input_var], test_prediction)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_err = 0
        batch = 5
        i = 0

        while i < X_train.shape[0] - batch + 1:
            # print (X_train[i:batch + i, ...].shape)
            train_err += train_fn(X_train[i:batch + i, ...], Y_train[i:batch + i])
            i += batch

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        print("  training loss:" + str(train_err))

    # full pass over test data

    val_err = 0
    val_acc = 0
    batch = 5
    i = 0
    # And a full pass over the validation data:
    while i < X_test.shape[0] - batch + 1:
        # print (X_val[i:batch + i, ...].shape)
        err, acc = val_fn(X_test[i:batch + i, ...], Y_test[i:batch + i])
        i += batch
        val_err += err
        val_acc += acc

    print("Final results:")
    print("  test loss:" + str(err))
    print("  test accuracy:" + str(acc * 100))

    # predic = predict(X_test)
    # print (predic)
    # print (predic.shape)
    # print (T.argmax(predic, axis=1).eval())
    # print (Y_test)
    # print (Y_test.shape)

   # print (T.mean(T.eq(T.argmax(predic, axis=1), Y_test),
    #              dtype=theano.config.floatX).eval())


if __name__ == '__main__':
    main(num_epochs=500)
