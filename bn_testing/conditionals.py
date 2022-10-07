from bn_testing.transformations.polynomials import RandomPolynomial
import pymc as pm


class Conditional(object):
    """
    """

    def make_transform(self, parents):
        """
        List of pymc distributions
        """
        raise NotImplementedError()

    def init(self, random):
        self.random = random

    def make_noise(self):
        return pm.Normal.dist(mu=0, sigma=0.1)

    def make(self, parents):
        """
        """
        return self.make_transform(parents) + self.make_noise()


class LinearConditional(Conditional):

    def make_transform(self, parents):
        # TODO: Random coefficients
        return sum(parents)
