import numpy
import theano
import theano.tensor as T
rng = numpy.random

N = 4
feats = 7
# astype(theano.config.floatX) is used for downcast to use float32 in GPU
D = (rng.randn(N, feats).astype(theano.config.floatX), rng.randint(size=N, low=0, high=2).astype(theano.config.floatX))
print "Shape of D0 " + str(D[0].shape)  # (400, 784)
print "Shape of D0 " + str(D[1].shape)  # (400,)
print D[0]
training_steps = 5

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
print w
b = theano.shared(0., name="b")
print b
print("Initial model:")
print(w.get_value())
print(b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)  # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()  # The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x, y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])
    print pred

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))
print('Diff')
print D[1] - predict(D[0])
a = D[0]
#print a
c = w.get_value()
#print c
print T.dot(a[0, :], c).eval()
