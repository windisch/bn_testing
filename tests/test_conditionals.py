import unittest
import pandas as pd
import numpy as np

from bn_testing.conditionals import PolynomialModel


class TestModel(unittest.TestCase):

    def test_sample(self):

        model = PolynomialModel(
            parents=['A', 'B', 'C'],
            random=np.random.RandomState(10),
        )

        df = pd.DataFrame(
            data={
                'A': np.random.normal(size=100),
                'B': np.random.normal(size=100),
                'C': np.random.normal(size=100),
                'D': np.random.normal(size=100),
            }
        )

        Y = model.sample(df)
        self.assertTupleEqual(Y.shape, (100,))
