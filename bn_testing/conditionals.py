import numpy as np
import pymc as pm

from bn_testing.helpers import sigmoid


class Conditional(object):
    """
    Base class of conditional distributions
    """

    def make_transform(self, parents):
        """
        Args:
            parents(list): List of :py:class:`pymc.Distribution` objects that represent the
            conditional variables

        Returns:
            pymc.Distribution: The conditional distribution
        """
        raise NotImplementedError()

    def init(self, random):
        self.random = random

    def make_noise(self):
        return pm.Normal.dist(mu=0, sigma=0.1)

    def __call__(self, parents):
        """
        """
        return self.make_transform(parents) + self.make_noise()


class LinearConditional(Conditional):
    """
    Linear conditionals
    """

    def __init__(self, coef_min=0.5, coef_max=1):
        self.coef_min = coef_min
        self.coef_max = coef_max

    def make_transform(self, parents):
        coefs = self.random.uniform(self.coef_min, self.coef_max, size=len(parents))
        signs = self.random.choice([-1, 1], size=len(parents))
        parents = np.array(parents)

        return np.sum(parents*coefs*signs)


class PolynomialConditional(Conditional):
    """
    Polynomial conditionals
    """

    def __init__(self, min_terms=1, max_terms=2, max_degree_add=10):
        self.min_terms = min_terms
        self.max_terms = max_terms
        self.max_degree_add = max_degree_add

    def _get_random_exponent(self, degree, n_variables):
        """
        Computes a random monomial exponent in `n_variables` many variables of given degree where
        no variable vanishes.

        Args:
            degree (int): Sum of all exponents
            n_variables (int): Number of variables

        Returns:
            numpy.ndarray: Integer Array of length `n_variables` that sum to `degree` where no entry
            is zero.

        """
        assert degree >= n_variables
        return np.ones(n_variables) + self.random.multinomial(
            degree - n_variables,
            [1/n_variables]*n_variables
        )

    def make_transform(self, parents):

        n_monomials = self.random.randint(self.min_terms, self.max_terms)
        degree = len(parents)+self.random.randint(1, self.max_degree_add)

        exponents = [
            self._get_random_exponent(
                degree=degree,
                n_variables=len(parents),
            ) for _ in range(n_monomials)
        ]

        signs = self.random.choice([-1, 1], size=n_monomials)
        coefs = signs * self.random.uniform(1, 10, size=n_monomials)

        return np.sum([
            sigmoid(
                coef*np.prod(np.power(parents, exp))
            ) for coef, exp in zip(coefs, exponents)
        ])
