import numpy as np
from bn_testing.helpers import sigmoid


class Transformation(object):

    def apply(self, parents):
        raise NotImplementedError()


class Linear(Transformation):

    def __init__(self, coefs):
        self.coefs = coefs

    def apply(self, parents):
        parents = np.array(parents)
        return np.sum(parents*self.coefs)


class Polynomial(Transformation):

    def __init__(self, exponents, coefs):
        self.exponents = exponents
        self.coefs = coefs

    def apply(self, parents):
        return np.sum([
            sigmoid(
                coef*np.prod(np.power(parents, exp))
            ) for coef, exp in zip(self.coefs, self.exponents)
        ])
