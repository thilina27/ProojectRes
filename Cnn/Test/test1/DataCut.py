import os
import cv2 as cv2
import random
import numpy as np

__author__ = 'thilina'


def load_Imagedata(imagedir='/home/thilina/Pictures/images', numChannels=1, minNumSamplesPerClass=10, imsize=(28, 28), p_train=0.6,
                   p_val=0.2):
    print("load images")
    # run throug the folders, load the files
    onlyFolders = [f for f in os.listdir(imagedir) if os.path.isdir(os.path.join(imagedir, f))]


    # Remove folders with too few samples
    for i in reversed(range(len(onlyFolders))):
        # get files in folder
        onlyFiles = [f for f in os.listdir(imagedir + '/' + onlyFolders[i]) if
                     os.path.isfile(os.path.join(imagedir + '/' + onlyFolders[i], f))]

        if len(onlyFiles) < minNumSamplesPerClass:
            del onlyFolders[i]

    # Run through all the folders
    for i in range(len(onlyFolders)):
        # get files in folder
        onlyFiles = [f for f in os.listdir(imagedir + '/' + onlyFolders[i]) if
                     os.path.isfile(os.path.join(imagedir + '/' + onlyFolders[i], f))]

        print(str(len(onlyFiles)) + " images of " + onlyFolders[i])

        # load the images one-by-one
        for ii in range(len(onlyFiles)):
            Imgtmp = cv2.imread(imagedir + '/' + onlyFolders[i] + '/' + onlyFiles[ii])

            # Convert to gray-scale
            if (numChannels == 1):
                if Imgtmp.shape[2] == 3:
                    Imgtmp = cv2.cvtColor(Imgtmp, cv2.COLOR_BGR2GRAY)
                Imgtmp = np.expand_dims(Imgtmp, axis=3)

            # resize image
            Imgtmpresize = cv2.resize(Imgtmp, (imsize[0], imsize[1]))

            # Based on the probabilities p_train, p_val: add to training, validation and test-set
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

            elif p_train <= rs < (p_val + p_train):
                if not 'X_val' in locals():
                    X_val = Imgtmpresize[None, ...]
                else:
                    X_val = np.concatenate((X_val, Imgtmpresize[None, ...]), axis=0)
                if not 'targets_val' in locals():
                    targets_val = np.array([i])
                else:
                    targets_val = np.concatenate((targets_val, np.array([i])))

            else:
                if not 'X_test' in locals():
                    X_test = Imgtmpresize[None, ...]
                else:
                    X_test = np.concatenate((X_test, Imgtmpresize[None, ...]), axis=0)
                if not 'targets_test' in locals():
                    targets_test = np.array([i])
                else:
                    targets_test = np.concatenate((targets_test, np.array([i])))

    if not 'targets_train' in locals():
        X_test = np.array(0, ndmin=3)
        targets_train = np.array(0)
    if not 'targets_val' in locals():
        X_val = np.array(0, ndmin=3)
        targets_val = np.array(0)
    if not 'targets_test' in locals():
        X_test = np.array(0, ndmin=3)
        targets_test = np.array(0)

    # typecast targets
    targets_test = targets_test.astype(np.int32)
    targets_val = targets_val.astype(np.int32)
    targets_train = targets_train.astype(np.int32)


    # apply some very simple normalization to the data
    X_test = X_test.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_train = X_train.astype(np.float32)

    X_test -= X_test.mean()
    X_test /= X_test.std()

    X_val -= X_val.mean()
    X_val /= X_val.std()

    X_train -= X_train.mean()
    X_train /= X_train.std()


    # permute dimensions.
    # The data has to have the layer dimension before the (x,y) dimension, as the conv. filters are applied to each layer and expect them to be in that order
    # The shape convention: (examples, channels, rows, columns)
    if numChannels == 1:  # add channel dimension if image is grayscale
        X_test = np.expand_dims(X_test, axis=4)
        X_val = np.expand_dims(X_val, axis=4)
        X_train = np.expand_dims(X_train, axis=4)

        X_test = np.transpose(X_test, (0, 3, 1, 2))
        X_val = np.transpose(X_val, (0, 3, 1, 2))
        X_train = np.transpose(X_train, (0, 3, 1, 2))

    return X_train, targets_train, X_val, targets_val, X_test, targets_test

if __name__ == '__main__':
    X_train1, targets_train1, X_val1, targets_val1, X_test1, targets_test1 = load_Imagedata();
    print(targets_val1);