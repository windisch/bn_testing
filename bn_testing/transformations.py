import numpy as np
import pymc as pm


class Transformation(object):
    """
    TODO
    """

    def __init__(self, parents, node):
        self.parents = parents
        self.node = node

    def get_vars_from_dict(self, parent_dict):
        return np.array([parent_dict[p] for p in self.parents])

    def apply(self, parents_mapping):
        raise NotImplementedError()


class Linear(Transformation):
    """
    TODO
    """

    def __init__(self, parents, node, coefs):
        super(Linear, self).__init__(parents=parents, node=node)
        self.coefs = coefs

    def apply(self, parents_mapping):
        parents = self.get_vars_from_dict(parents_mapping)
        return np.sum(parents*self.coefs)

    def __repr__(self):
        return f"{self.node} = " + " + ".join(
            ["{:.1f}*{}".format(c, p) for c, p in zip(self.coefs, self.parents)]
        )


class Polynomial(Transformation):
    """
    TODO
    """

    def __init__(self, parents, node, exponents, coefs, with_tanh=True):
        Transformation.__init__(self, parents=parents, node=node)
        self.with_tanh = with_tanh
        self.exponents = np.array(exponents, dtype=int)
        self.coefs = coefs

    def _compute_monomial(self, parents, exp, coef):
        monomial = coef*np.prod(np.power(parents, exp))

        if self.with_tanh:
            return np.tanh(monomial)
        else:
            return monomial

    def apply(self, parents_mapping):
        parents = self.get_vars_from_dict(parents_mapping)
        return np.sum([
            self._compute_monomial(
                parents=parents,
                exp=exp,
                coef=coef
            ) for coef, exp in zip(self.coefs, self.exponents)
            ]
        )

    def __repr__(self):
        return f"{self.node} = " + " + ".join(
            ["{:.1f}*{}".format(
                c,
                "*".join([f"{p}^{e}" for p, e in zip(self.parents, exp.ravel().astype(str).tolist())])
            ) for c, exp in zip(self.coefs, self.exponents)]
        )


class Constant(Transformation):
    """
    TODO
    """

    def __init__(self, parents, node, value):
        Transformation.__init__(self, parents=parents, node=node)
        self.value = value

    def apply(self, parents_mapping):
        return pm.math.constant(self.value)
