from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

__author__ = 'thilina'


srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

print f()
print f()

print g()
print g()