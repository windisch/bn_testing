import unittest
from unittest.mock import MagicMock
import numpy as np
import pymc as pm
from scipy.stats import normaltest

from bn_testing.conditionals import (
    LinearConditional,
    PolynomialConditional,
)


class TestLinearConditionals(unittest.TestCase):

    def setUp(self):
        self.x = pm.Normal.dist(mu=0, sigma=1)
        self.y = pm.Beta.dist(alpha=2, beta=2)
        self.conditional = LinearConditional()

        random = MagicMock()
        random.uniform = MagicMock(return_value=[1, 1])
        random.choice = MagicMock(return_value=[1, 1])

        self.conditional.init(random)

    def test_transform(self):
        self.conditional.random = MagicMock()
        self.conditional.random.uniform = MagicMock(return_value=[1, 1])
        self.conditional.random.choice = MagicMock(return_value=[1, 1])

        z = self.conditional.make_transform([self.x, self.y])

        X, Y, Z = pm.draw([self.x, self.y, z], 100)
        np.testing.assert_array_almost_equal(
            Z,
            X+Y
        )

    def test_make(self):
        z = self.conditional.make([self.x, self.y])
        X, Y, Z = pm.draw([self.x, self.y, z], 1000)
        N = Z - (X+Y)
        _, p = normaltest(N)
        self.assertGreater(p, 1e-3)


class TestPolynomialConditionals(unittest.TestCase):

    def setUp(self):
        self.x = pm.Normal.dist(mu=0, sigma=1)
        self.y = pm.Beta.dist(alpha=2, beta=2)
        self.z = pm.Beta.dist(alpha=1, beta=1)
        self.conditional = PolynomialConditional()
        self.conditional.init(np.random.RandomState(10))

    def test_all_variables_occur_in_monomials(self):
        for _ in range(100):
            exp = self.conditional._get_random_exponent(10, 5)
            self.assertEqual(exp.sum(), 10)
            self.assertTrue(np.all(exp >= 1))
