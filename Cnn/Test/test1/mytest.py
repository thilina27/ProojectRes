from __future__ import print_function
from PIL import Image

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

__author__ = 'thilina'

# open random image of dimensions 639x516
img = Image.open(open('/home/thilina/Pictures/images.jpg'))
# dimensions are (height, width, channel)
img = np.asarray(img, dtype='float32') / 256.
print(img.shape)
# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 95, 127)

print(img_.shape)

print(img_)
network = lasagne.layers.InputLayer(shape=(None, 3, 95, 127), input_var=img_)
print(network.shape)
network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
print(network.filter_size)

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

prediction = lasagne.layers.get_output(network)

print(prediction)

print("=================================")

target = "1000"
target_var = T.ivector(target)

loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

print(loss)

print("=================================")