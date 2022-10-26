import numpy as np
import pymc as pm

from bn_testing.terms import (
    Linear,
    Polynomial,
    Constant,
)


class Conditional(object):
    """
    Base class of conditional distributions
    """

    def make_term(self, parents, node):
        """
        Builds a term randomly.


        :param list parents: Name of the parent nodes.
        :param str node: Name of the node whose term  should be made

        :returns: A term
        :rtype: bn_testing.term.Term
        """
        raise NotImplementedError()

    def init(self, random):
        self.random = random

    def make_noise(self):
        return pm.Normal.dist(mu=0, sigma=0.05)

    def make_source(self):
        return pm.Beta.dist(
            alpha=self.random.uniform(1, 5),
            beta=self.random.uniform(1, 5),
        )

    def __call__(self, parents, node):
        """
        """
        return self.make_term(parents=parents, node=node)


class LinearConditional(Conditional):
    """
    Linear conditionals
    """

    def __init__(self, coef_min=0.5, coef_max=1):
        self.coef_min = coef_min
        self.coef_max = coef_max

    def make_term(self, parents, node):
        n_parents = len(parents)
        signs = self.random.choice([-1, 1], size=n_parents)
        coefs = signs*self.random.uniform(self.coef_min, self.coef_max, size=n_parents)
        return Linear(
            parents=parents,
            coefs=coefs
        )


class PolynomialConditional(Conditional):
    """
    Conditional that builds polynomial terms
    """

    def __init__(self, min_terms=1, max_terms=5, max_degree_add=10, with_tanh=True):
        self.min_terms = min_terms
        self.max_terms = max_terms
        self.max_degree_add = max_degree_add
        self.with_tanh = with_tanh

    def _get_random_exponent(self, degree, n_variables):
        """
        Computes a random monomial exponent in `n_variables` many variables of given degree where
        no variable vanishes.

        :param int degree: Sum of all exponents
        :param int n_variables: Number of variables

        :returns: Integer Array of length `n_variables` that sum to `degree` where no entry
            is zero.
        :rtype: numpy.ndarray

        """
        assert degree >= n_variables
        return np.ones(n_variables) + self.random.multinomial(
            degree - n_variables,
            [1/n_variables]*n_variables
        )

    def make_term(self, parents, node):
        n_parents = len(parents)

        n_monomials = self.random.randint(self.min_terms, self.max_terms+1)
        degree = n_parents+self.random.randint(1, self.max_degree_add)

        exponents = [
            self._get_random_exponent(
                degree=degree,
                n_variables=n_parents,
            ) for _ in range(n_monomials)
        ]

        signs = self.random.choice([-1, 1], size=n_monomials)
        coefs = signs * self.random.uniform(1, 10, size=n_monomials)
        return Polynomial(
            parents=parents,
            exponents=exponents,
            coefs=coefs,
            intercept=0,
            with_tanh=self.with_tanh,
        )


class ConstantConditional(Conditional):
    """
    A conditional yielding constant values. Used in intervensions and to compute causal effects.
    """

    def __init__(self, value):
        self.value = value

    def make_term(self, parents, node):
        return Constant(parents=parents, value=self.value)

    def make_noise(self):
        return pm.math.constant(0)

    def make_source(self):
        return pm.math.constant(self.value)
