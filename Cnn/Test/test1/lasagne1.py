from __future__ import print_function
import gzip

import numpy as np
import os
import sys
import theano
import theano.tensor as T

import lasagne
import random
import cv2
import time

__author__ = 'thilina'


def loadMnist():
    def load_mnist_images(filename):
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def readData():
    imagedir = '/home/thilina/Data/test1'
    p_train = 0.5

    onlyFolders = [f for f in os.listdir(imagedir) if os.path.isdir(os.path.join(imagedir, f))]

    for i in reversed(range(len(onlyFolders))):
        # get files in folder
        onlyFiles = [f for f in os.listdir(imagedir + '/' + onlyFolders[i]) if
                     os.path.isfile(os.path.join(imagedir + '/' + onlyFolders[i], f))]
        print(str(len(onlyFiles)) + " images of " + onlyFolders[i])

        for ii in range(len(onlyFiles)):
            Imgtmp = cv2.imread(imagedir + '/' + onlyFolders[i] + '/' + onlyFiles[ii])
            if Imgtmp.shape[2] == 3:
                Imgtmp = cv2.cvtColor(Imgtmp, cv2.COLOR_BGR2GRAY)
            # resize image
            Imgtmp = cv2.resize(Imgtmp, (50, 50))
            Imgtmpresize = np.expand_dims(Imgtmp, axis=3)

            rs = random.random()
            if rs < p_train:
                if not 'X_train' in locals():
                    X_train = Imgtmpresize[None, ...]
                else:
                    X_train = np.concatenate((X_train, Imgtmpresize[None, ...]), axis=0)
                if not 'targets_train' in locals():
                    targets_train = np.array([i])
                else:
                    targets_train = np.concatenate((targets_train, np.array([i])))

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


    # apply some very simple normalization to the data
    X_test = X_test.astype(np.float32)
    X_train = X_train.astype(np.float32)

    X_test = np.transpose(X_test, (0, 3, 1, 2))
    X_train = np.transpose(X_train, (0, 3, 1, 2))

    return X_train, Y_train, X_test, Y_test


def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 50, 50),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
        l_in_drop, num_units=800,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
        l_hid1_drop, num_units=800,
        nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
        l_hid2_drop, num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


def main(model='cnn', num_epochs=2):
    # Load the dataset
    print("Loading data...")
    # X_train, Y_train, X_test, Y_test = readData()
    X_train, y_train, X_val, y_val, X_test, y_test = loadMnist()

    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)
    print(y_test.shape)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
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

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        start_time = time.time()

        train_err = 0
        batch = 1000
        i = 0

        print (y_train[0:5])

        while i < 50000 - batch + 1:
            # print (X_train[i:batch + i, ...].shape)

            train_err += train_fn(X_train[i:batch + i, ...], y_train[i:batch + i])
            i += batch

        val_err = 0
        val_acc = 0
        batch = 1000
        i = 0
        # And a full pass over the validation data:
        while i < 10000 - batch + 1:
            # print (X_val[i:batch + i, ...].shape)
            err, acc = val_fn(X_val[i:batch + i, ...], y_val[i:batch + i])
            i += batch
            val_err += err
            val_acc += acc



        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err))
        print("  validation loss:\t\t{:.6f}".format(val_err))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc))

    # After training, we compute and print the test error:
    err = 0
    acc = 0
    batch = 1000
    i = 0
    while i < 10000 - batch + 1:
        err, acc = val_fn(X_test[i:batch + i, ...], y_test[i:batch + i])
        err += err
        acc += acc
        i += batch

    print("Final results:")
    print("  test loss:" + str(err))
    print("  test accuracy:" + str(acc))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)
