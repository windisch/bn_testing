import numpy as np
from bn_testing.transformations.polynomials import RandomPolynomial


class NodeModel(object):

    def sample(self, df):
        raise NotImplementedError()


class PolynomialModel(NodeModel):
    """
    """

    def __init__(self, parents, random=None, sigma=0.02):

        self.parents = parents
        self.random = random

        self.sigma = sigma

        self.polynomial = RandomPolynomial(
            n_monomials=len(self.parents),
            degree=len(self.parents)+2,
            n_variables=len(self.parents),
            random=self.random
        )

    def _transform(self, X):
        return self.polynomial.apply(X)

    def _get_noise(self, n):
        return self.random.normal(
            loc=0,
            size=n,
            scale=self.sigma)

    def sample(self, df):
        """
        Samples for each row of the :py:cls:`pandas.DataFrame` an element from the conditional
        distribution.
        """
        X = df[self.parents].values
        return self._transform(X) + self._get_noise(df.shape[0])
