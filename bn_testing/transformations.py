import numpy as np
from bn_testing.helpers import sigmoid


class Transformation(object):

    def __init__(self, parents):
        self.parents = parents

    def get_vars_from_dict(self, parent_dict):
        return np.array([parent_dict[p] for p in self.parents])

    def apply(self, parents_mapping):
        raise NotImplementedError()


class Linear(Transformation):

    def __init__(self, parents, coefs):
        super(Linear, self).__init__(parents)
        self.coefs = coefs

    def apply(self, parents_mapping):
        parents = self.get_vars_from_dict(parents_mapping)
        return np.sum(parents*self.coefs)

    def __repr__(self):
        return " + ".join(
            ["{:.1f}*{}".format(c, p) for c, p in zip(self.coefs, self.parents)]
        )


class Polynomial(Transformation):

    def __init__(self, parents, exponents, coefs):
        Transformation.__init__(self, parents)
        self.exponents = np.array(exponents, dtype=int)
        self.coefs = coefs

    def apply(self, parents_mapping):
        parents = self.get_vars_from_dict(parents_mapping)
        return np.sum([
            sigmoid(
                coef*np.prod(np.power(parents, exp))
            ) for coef, exp in zip(self.coefs, self.exponents)
        ])

    def __repr__(self):
        return " + ".join(
            ["{:.1f}*{}".format(
                c,
                "*".join([f"{p}^{e}" for p, e in zip(self.parents, exp.ravel().astype(str).tolist())])
            ) for c, exp in zip(self.coefs, self.exponents)]
        )


class Constant(Transformation):

    def __init__(self, parents, value):
        super(Constant, self).__init__(parents)
        self.value = value

    def apply(self, parents_mapping):
        return self.value
