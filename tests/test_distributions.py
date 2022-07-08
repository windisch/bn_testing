import unittest
import pandas as pd
import numpy as np

from bn_testing.models import ConditionalGaussian


class TestModel(unittest.TestCase):

    def test_sample(self):

        model = ConditionalGaussian(
            parents=['A', 'B', 'C']
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
