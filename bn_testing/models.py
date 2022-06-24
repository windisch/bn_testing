import numpy as np


class Model(object):

    def sample(self, df):
        raise NotImplementedError()


class ConditionalGaussian(Model):

    def __init__(self, parents, random_state=10):

        self.parents = parents

        self.random = np.random.RandomState(random_state)

        self.weights = np.random.uniform(
            low=-1,
            high=1,
            size=len(self.parents)
        )
        self.sigma = 0.1

    def sample(self, df):

        X = df[self.parents].values

        # TODO: Allow more transformations
        means = np.matmul(X, self.weights)

        return self.random.normal(
            loc=means,
            scale=self.sigma)
