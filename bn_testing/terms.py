import numpy as np
import pymc as pm
import numbers

from bn_testing.helpers import sigmoid


class Term(object):
    """
    A mathematical expression that transforms the parent nodes into the given node

    :param list parents: List of parent nodes
    :param function term_fn: Function that maps a :py:class:`dict` where the parent names are the
        keys and pymc variables are the values to a pymc variable
    :param str disp: String that should be shown when the term is displayed
    """

    def __init__(self, parents, term_fn=None, disp=""):
        self.parents = parents
        self.disp = disp

        if term_fn is None:
            self.term_fn = lambda x: pm.math.constant(0)
        else:
            self.term_fn = term_fn

    def get_vars_from_dict(self, parent_dict):
        """
        Flattens the dict of parents to a list with order specified in the constructor.

        :param parent_dict: :py:class:`dict` where the keys are parent names and the values are
            variables.

        :returns: List of parent variables
        :rtype: list
        """
        return np.array([parent_dict[p] for p in self.parents])

    def apply(self, parents_mapping):
        """
        Applies the term onto the parents
        """
        return self.term_fn(parents_mapping)

    def __repr__(self):
        return self.disp

    def _add_with(self, term):
        def term_fn(parents_mapping):
            return self.apply(parents_mapping) + term.apply(parents_mapping)
        return term_fn

    def _multiply_with(self, term):
        def term_fn(parents_mapping):
            return self.apply(parents_mapping) * term.apply(parents_mapping)
        return term_fn

    def _make_constant_term(self, value):
        return Constant(
            parents=self.parents,
            value=value,
        )

    def _power(self, k):
        def term_fn(parents_mapping):
            return np.power(self.apply(parents_mapping), k)
        return term_fn

    def __pow__(self, k):
        return Term(
            parents=self.parents,
            term_fn=self._power(k),
            disp=f"({self.disp})^{k}"
        )

    def __radd__(self, value):
        return self._make_constant_term(value) + self

    def __ladd__(self, value):
        return self + self._make_constant_term(value)

    def __add__(self, term):
        return Term(
            parents=self.parents,
            term_fn=self._add_with(term),
            disp=f"{self.disp}+{term.disp}"
        )

    def __lmul__(self, value):
        return self._make_constant_term(value) * self

    def __rmul__(self, value):
        return self*self._make_constant_term(value)

    def __mul__(self, term):

        if isinstance(term, numbers.Number):
            term = self._make_constant_term(term)

        return Term(
            parents=self.parents,
            term_fn=self._multiply_with(term),
            disp=f"({self.disp})*({term.disp})"
        )


class Linear(Term):
    """
    A linear weighted sum of the parent variables

    :param list parents: List of parent nodes
    :param numpy.ndarray coefs: Array of coefficients, size must equal the number of parents
    """

    def __init__(self, parents, coefs):
        self.coefs = coefs
        super(Linear, self).__init__(
            parents=parents,
            disp=" + ".join(
                ["{:.1f}*{}".format(c, p) for c, p in zip(self.coefs, parents)]
            )
        )

    def apply(self, parents_mapping):
        parents = self.get_vars_from_dict(parents_mapping)
        return np.sum(parents*self.coefs)


class Polynomial(Term):
    """
    A multivariate polynomial in the parents variables.

    :param list parents: List of parent nodes
    :param numpy.ndarray exponents: Array holding the exponents of the parent variables. Values can
        be negative.
    :param numpy.ndarray coefs: Array holding the coeficients for each monomial
    :param float intercept: The intercept of the polynomial
    :param bool with_tanh: Whether :py:func:`numpy.tanh` should be applied onto the monmial
    """
    def __init__(self, parents, exponents, coefs, intercept=0, with_tanh=False):
        self.exponents = np.array(exponents)
        self.coefs = np.array(coefs)
        self.with_tanh = with_tanh
        self.intercept = intercept

        if len(parents) != self.exponents.shape[1]:
            raise ValueError('Exponents do not match parents')

        if self.coefs.shape[0] != self.exponents.shape[0]:
            raise ValueError('Exponents do not match coefs')

        super(Polynomial, self).__init__(
            parents=parents,
            disp="+".join(
                [
                   Polynomial._disp_term(
                       parents=parents,
                       exp=e,
                       coef=c
                   ) for e, c in zip(self.exponents, self.coefs)
                ]
            )
        )

    @staticmethod
    def _disp_term(parents, exp, coef):
        return "{:.1f}*".format(coef) + "*".join(
            [
                f"{p}^{e}" for p, e in zip(parents, exp.astype(str).tolist())
            ]
        )

    def _get_monomial(self, parents, exp):
        monomial = np.prod(np.power(parents, exp))

        if self.with_tanh:
            return sigmoid(monomial)
        else:
            return monomial

    def apply(self, parents_mapping):
        parents = self.get_vars_from_dict(parents_mapping)
        monomials = [self._get_monomial(parents, exp) for exp in self.exponents]
        return sum([c*m for c, m in zip(self.coefs, monomials)])+self.intercept


class Constant(Term):
    """
    A constant term.
    """

    def __init__(self, parents, value):
        super(Constant, self).__init__(
            parents=parents,
            term_fn=lambda _: pm.math.constant(value),
            disp="{:.1f}".format(value)
        )
