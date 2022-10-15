"""
TODO
"""
import numpy as np
import pymc as pm
import numbers

from bn_testing.helpers import sigmoid


class Term(object):
    """
    TODO
    """

    def __init__(self, parents, node, term_fn=None, disp=""):
        self.parents = parents
        self.node = node
        self.disp = disp
        # TODO: Catch disp="" case

        if term_fn is None:
            self.term_fn = lambda x: 0
        else:
            self.term_fn = term_fn

    def get_vars_from_dict(self, parent_dict):
        return np.array([parent_dict[p] for p in self.parents])

    def apply(self, parents_mapping):
        return self.term_fn(parents_mapping)

    def __repr__(self):
        return f"{self.node} = {self.disp}"

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
            node=self.node,
            value=value,
        )

    def _power(self, k):
        def term_fn(parents_mapping):
            return np.power(self.apply(parents_mapping), k)
        return term_fn

    def __pow__(self, k):
        return Term(
            parents=self.parents,
            node=self.node,
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
            node=self.node,
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
            node=self.node,
            term_fn=self._multiply_with(term),
            disp=f"({self.disp})*({term.disp})"
        )


class Linear(Term):
    """
    TODO
    """

    def __init__(self, parents, node, coefs):
        self.coefs = coefs
        super(Linear, self).__init__(
            parents=parents,
            node=node,
            disp=" + ".join(
                ["{:.1f}*{}".format(c, p) for c, p in zip(self.coefs, parents)]
            )
        )

    def apply(self, parents_mapping):
        parents = self.get_vars_from_dict(parents_mapping)
        return np.sum(parents*self.coefs)


class Monomial(Term):
    """
    TODO
    """
    def __init__(self, parents, node, exponents, with_tanh=False):
        self.exponents = np.array(exponents).ravel()
        self.with_tanh = with_tanh

        if len(parents) != self.exponents.shape[0]:
            raise ValueError('Exponents do not match parents')

        super(Monomial, self).__init__(
            parents=parents,
            node=node,
            disp="*".join(
                [
                    f"{p}^{e}" for p, e in zip(parents, self.exponents.astype(str).tolist())
                ]
            )
        )

    def apply(self, parents_mapping):
        parents = self.get_vars_from_dict(parents_mapping)
        monomial = np.prod(np.power(parents, self.exponents))

        if self.with_tanh:
            return sigmoid(monomial)
        else:
            return monomial


class Constant(Term):
    """
    TODO
    """

    def __init__(self, parents, node, value):
        super(Constant, self).__init__(
            parents=parents,
            node=node,
            term_fn=lambda _: pm.math.constant(value),
            disp="{:.1f}".format(value)
        )
