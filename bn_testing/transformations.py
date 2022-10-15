"""
TODO
"""
import numpy as np
import pymc as pm
import numbers

from bn_testing.helpers import sigmoid


class Transformation(object):
    """
    TODO
    """

    def __init__(self, parents, node, transformation_fn=None, disp=""):
        self.parents = parents
        self.node = node
        self.disp = disp
        # TODO: Catch disp="" case

        if transformation_fn is None:
            self.transformation_fn = lambda x: 0
        else:
            self.transformation_fn = transformation_fn

    def get_vars_from_dict(self, parent_dict):
        return np.array([parent_dict[p] for p in self.parents])

    def apply(self, parents_mapping):
        return self.transformation_fn(parents_mapping)

    def __repr__(self):
        return f"{self.node} = {self.disp}"

    def _add_with(self, transformation):
        def transformation_fn(parents_mapping):
            return self.apply(parents_mapping) + transformation.apply(parents_mapping)
        return transformation_fn

    def _multiply_with(self, transformation):
        def transformation_fn(parents_mapping):
            return self.apply(parents_mapping) * transformation.apply(parents_mapping)
        return transformation_fn

    def _make_constant_transformation(self, value):
        return Constant(
            parents=self.parents,
            node=self.node,
            value=value,
        )

    def _power(self, k):
        def transformation_fn(parents_mapping):
            return np.power(self.apply(parents_mapping), k)
        return transformation_fn

    def __pow__(self, k):
        return Transformation(
            parents=self.parents,
            node=self.node,
            transformation_fn=self._power(k),
            disp=f"({self.disp})^{k}"
        )

    def __radd__(self, value):
        return self._make_constant_transformation(value) + self

    def __ladd__(self, value):
        return self + self._make_constant_transformation(value)

    def __add__(self, transformation):
        return Transformation(
            parents=self.parents,
            node=self.node,
            transformation_fn=self._add_with(transformation),
            disp=f"{self.disp}+{transformation.disp}"
        )

    def __lmul__(self, value):
        return self._make_constant_transformation(value) * self

    def __rmul__(self, value):
        return self*self._make_constant_transformation(value)

    def __mul__(self, transformation):

        if isinstance(transformation, numbers.Number):
            transformation = self._make_constant_transformation(transformation)

        return Transformation(
            parents=self.parents,
            node=self.node,
            transformation_fn=self._multiply_with(transformation),
            disp=f"({self.disp})*({transformation.disp})"
        )


class NumpyFunc(Transformation):

    def __init__(self, node, func, parent='input'):

        self.parent = parent

        super(Transformation, self).__init__(
            parents=[self.parent],
            node=node,
            disp=f"{str(func)}({self.parent})",
            transformation_fn=lambda mapping: func(mapping[self.parent])
        )


class Linear(Transformation):
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


class Monomial(Transformation):
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


class Constant(Transformation):
    """
    TODO
    """

    def __init__(self, parents, node, value):
        Transformation.__init__(
            self,
            parents=parents,
            node=node,
            transformation_fn =lambda _: pm.math.constant(value),
            disp="{:.1f}".format(value)
        )
