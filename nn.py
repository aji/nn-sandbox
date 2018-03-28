import numpy as np
from collections import defaultdict

class NetworkUnit(object):
    '''
    base class for units in a gradient descent network

    w is the unit's learnable parameters
    x is the input to the unit
    y is the output from the unit
    z is data shared from forward to backward propagation
    dX is cost gradient with respect to X
    Xs is a collection of X

    forward(x) computes y, z
    backward(x, z, dy) computes dx, dw
    train(dws, learn) performs gradient descent with the given learn rate
    '''

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x, z, dy):
        raise NotImplementedError()

    def train(self, dws, learn):
        raise NotImplementedError()

class Linear(NetworkUnit):
    ''' linear transformation '''

    def __init__(self, insize, outsize):
        self._w = 0.01 * (np.random.rand(outsize, insize) - 0.5)

    def forward(self, x):
        return np.dot(self._w, x), None

    def backward(self, x, z, dy):
        return np.dot(self._w.T, dy), np.outer(dy, x)

    def train(self, dws, learn):
        self._w -= np.clip(sum(dws), -5., 5.) * learn

class LeakyReLU(NetworkUnit):
    ''' leaky ReLU '''

    def __init__(self, size):
        self._b = np.zeros(size)

    def forward(self, x):
        z = x + self._b
        return np.where(z > 0., z, 0.01 * z), z

    def backward(self, x, z, dy):
        dx = db = dy * np.where(z > 0., 1., 0.01)
        return dx, db

    def train(self, dws, learn):
        self._b -= np.clip(sum(dws), -5., 5.) * learn

class Tanh(NetworkUnit):
    ''' hyperbolic tangent '''

    def __init__(self, size):
        self._b = np.zeros(size)

    def forward(self, x):
        z = x + self._b
        return np.tanh(z), z

    def backward(self, x, z, dy):
        dx = db = dy * (1. - np.tanh(z) ** 2.)
        return dx, db

    def train(self, dws, learn):
        self._b -= np.clip(sum(dws), -5., 5.) * learn

class CostFunction(object):
    def cost(self, label, y):
        raise NotImplementedError()

class SumOfSquares(CostFunction):
    def cost(self, label, y):
        d = y - label
        return d ** 2., 2. * d

class Layered(object):
    def __init__(self, costfn, learn):
        self._layers = []
        self._costfn = costfn
        self._learn = learn

    def add(self, layer):
        self._layers.append((len(self._layers), layer))

    def train(self, examples):
        xs = dict()
        zs = dict()
        dws = defaultdict(list)

        total_cost = 0.

        for example, label in examples:
            xs[0] = example

            for i, layer in self._layers:
                xs[i + 1], zs[i] = layer.forward(xs[i])

            cost, dy = self._costfn.cost(label, xs[len(self._layers)])
            total_cost += sum(cost)

            for i, layer in reversed(self._layers):
                dy, dw = layer.backward(xs[i], zs[i], dy)
                dws[i].append(dw)

        for i, layer in self._layers:
            layer.train(dws[i], self._learn)

        return total_cost / len(examples)

    def sample(self, x):
        for i, layer in self._layers:
            x, _ = layer.forward(x)
        return x
