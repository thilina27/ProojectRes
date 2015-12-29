import numpy as np

__author__ = 'thilina'

a = np.arange(10) ** 3
print a
# from start to position 6 2 skip positions
print a[:6:2]
# reversed
print a[::-1]

for i in a:
    print(i ** (1 / 3.0))

c = np.array([[[0, 1, 2],  # a 3D array (two stacked 2D arrays)
               [10, 12, 13]],
              [[100, 101, 102],
               [110, 112, 113]]])
print c.shape

print c[1, ...]  # same as c[1,:,:] or c[1]

print c[..., 2]  # same as c[:,:,2]

# iterate over 1st axis
for row in c:
    print(row)
    print ' 1 '

# if need to go trough each element we can use flat
for element in c.flat:
    print(element)


