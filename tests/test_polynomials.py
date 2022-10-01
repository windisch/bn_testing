import unittest
import numpy as np

from bn_testing.transformations.polynomials import (
    _get_random_exponent,
    RandomPolynomial,
)


class TestRandomExponent(unittest.TestCase):

    def test_get_random_exponent(self):
        random = np.random.RandomState(11)

        exp = _get_random_exponent(10, 3, random)
        self.assertTrue(np.all(exp >=1))
        self.assertTrue(exp.sum(), 10)
        self.assertTupleEqual(exp.shape, (3,))


class TestRandomPolynomial(unittest.TestCase):

    def setUp(self):
        random = np.random.RandomState(10)
        self.poly = RandomPolynomial(
            n_monomials=4,
            degree=10,
            n_variables=5,
            random=random)

    def test_exponents(self):
        for exp in self.poly.exponents:
            self.assertTrue(exp.sum(), 10)

    def test_apply(self):
        X = np.random.normal(size=(100, 5))
        Y = self.poly.apply(X)
        self.assertTupleEqual(Y.shape, (X.shape[0],))
