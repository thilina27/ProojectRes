import numpy as np

__author__ = 'thilina'

a = np.floor(10 * np.random.random((3, 4)))
print a
print a.shape

print a.ravel()  # flatten the array
a.shape = (6, 2)  # change shape
print a
a = a.T  # transpose

print a

b = a.reshape(3, -1)

print b

# array can be stacked
a = np.floor(10 * np.random.random((2, 2)))
print a

b = np.floor(10 * np.random.random((2, 2)))
print b

print (np.vstack((a, b)))

print (np.hstack((a, b)))
