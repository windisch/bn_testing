import numpy as np


def _get_random_exponent(degree, n_variables, random=None):

    assert degree >= n_variables
    # Every variable appears
    exp = np.zeros(n_variables)
    for i in range(n_variables-1):
        exp[i] = exp[i] + random.randint(0, degree - n_variables - exp[:i].sum())

    exp = exp + 1
    exp[n_variables-1] = degree - np.sum(exp[:(n_variables-1)])
    return exp


def sigmoid(x):
    return 1/(1+np.exp(-x))


class RandomPolynomial(object):

    def __init__(self, n_monomials, degree, n_variables, random):

        self.exponents = [
            _get_random_exponent(degree, n_variables, random) for _ in range(n_monomials)
        ]
        self.signs = random.choice([-1, 1], size=n_monomials)
        self.coefs = self.signs * random.uniform(1, 10, size=n_monomials)

    def apply(self, X):
        XX = np.zeros(X.shape[0])
        for exp, coef in zip(self.exponents, self.coefs):
            XX = XX + coef*np.prod(np.sin(np.pi*X**exp), axis=1)
        return sigmoid(XX)
