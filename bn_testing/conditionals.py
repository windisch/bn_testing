import numpy as np


class ConditionalDistribution(object):

    def sample(self, df):
        raise NotImplementedError()


class ConditionalGaussian(ConditionalDistribution):
    """
    """

    def __init__(self, parents, random_state=10, sigma=0.1):

        self.parents = parents

        self.random = np.random.RandomState(random_state)

        self.weights = np.random.uniform(
            low=-1,
            high=1,
            size=len(self.parents)
        )
        self.sigma = sigma

    def _get_mean(self, X):
        return np.matmul(X, self.weights)

    def sample(self, df):
        """
        Samples for each row of the :py:cls:`pandas.DataFrame` an element from the conditional
        distribution.
        """
        X = df[self.parents].values
        return self.random.normal(
            loc=self._get_mean(X),
            scale=self.sigma)
