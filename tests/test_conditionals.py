import unittest
import numpy as np
import pymc as pm

from bn_testing.conditionals import (
    LinearConditional,
    PolynomialConditional,
)


class TestLinearConditionals(unittest.TestCase):

    def setUp(self):
        self.x = pm.Normal.dist(mu=0, sigma=1)
        self.y = pm.Beta.dist(alpha=2, beta=2)
        self.conditional = LinearConditional()
        self.conditional.init(np.random.RandomState(10))
        self.transformation = self.conditional(2)

    def test_transform(self):

        w_x = self.transformation.coefs[0]
        w_y = self.transformation.coefs[1]

        z = self.transformation.apply([self.x, self.y])

        X, Y, Z = pm.draw([self.x, self.y, z], 100)
        np.testing.assert_array_almost_equal(
            Z,
            w_x*X+w_y*Y
        )


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
