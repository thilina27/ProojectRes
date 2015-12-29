import numpy
import theano.tensor as T
from theano import function, shared
from theano import pp

__author__ = 'thilina'

x = T.dscalar('x')  # single values scalar of double
# http://deeplearning.net/software/theano/library/tensor/basic.html

y = T.dscalar('y')

print type(x)
print x.type

print T.dscalar

z = x + y
f = function([x, y], z)  # compile function in x,y out z

print f(1, 2)

print pp(z)  # show how z is created

# adding matrix
x = T.dmatrix('x')
y = T.dmatrix('y')

z = x + y

m = function([x, y], z)

print m([[1, 2, 3], [1, 2, 3]], [[1, 3, 4], [1, 3, 4]])

# exercise 1
a = T.vector()  # declare variable
b = T.vector()
# out = a + a ** 10               # build symbolic expression
out = a ** 2 + b ** 2 + 2 * a * b
f = function([a, b], out)  # compile function
print(f([0, 1, 2], [1, 2, 3]))

r = shared(numpy.random.randn(3, 4))
print r.get_value()
