import unittest
import pandas as pd
import numpy as np
import pymc as pm
from scipy.stats import normaltest

from bn_testing.conditionals import LinearConditional


class TestLinearConditionals(unittest.TestCase):

    def setUp(self):
        self.x = pm.Normal.dist(mu=0, sigma=1)
        self.y = pm.Beta.dist(alpha=2, beta=2)
        self.conditional = LinearConditional()

    def test_transform(self):
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
