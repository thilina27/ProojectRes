import numpy as np

a = np.arange(15).reshape(3, 5)

# print np array
print a
# print shape of the array
print a.shape
# print #of dimensions - rank
print a.ndim
# print data type
print a.dtype.name
# print item size
print a.itemsize
# print actual size (how many elements)
print a.size
# print array type
print type(a)

# Data type of array is choose automatically when different type of data given
a = np.array([2, 3, 4])
print a.dtype

b = np.array([1.2, 3.5, 5.1])
print b.dtype

# sequences of sequences into two-dimensional arrays
b = np.array([(1.5, 2, 3), (4, 5, 6)])
print b

# Data type can be also define at the initiation time
c = np.array([[1, 2], [3, 4]], dtype=complex)
print c.dtype.name

# can create array with zeros , ones , and randoms (data type is float 64)
np.zeros((3, 4))

np.ones((2, 3, 4), dtype=np.int16)  # dtype can also be specified

np.empty((2, 3))  # uninitialized, output may vary

# can create sequence of numbers (start , end , increment )
np.arange(10, 30, 5)

# 9 numbers from 0 to 2
np.linspace(0, 2, 9)

# array types
a = np.arange(6)  # 1d array
print(a)

b = np.arange(12).reshape(4, 3)  # 2d array
print(b)

c = np.arange(24).reshape(2, 3, 4)  # 3d array
print(c)


# if array is too large to display it will skip showing middle content
print(np.arange(10000).reshape(100, 100))
# If need to see all
# np.set_printoptions(threshold='nan')

# basic arithmetic operation will return a new array
a = np.array([20, 30, 40, 50])
b = np.arange(4)
print b

c = a - b
print c

print b ** 2

print 10 * np.sin(a)

print a < 35

# * will works on element wise if need to get matrix product use dot

A = np.array([[1, 1],
              [0, 1]])
B = np.array([[2, 0],
              [3, 4]])
print A * B  # element wise product

print A.dot(B)  # matrix product

print np.dot(A, B)  # another matrix product

