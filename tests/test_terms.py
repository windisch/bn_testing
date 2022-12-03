import unittest
import numpy as np
import pymc as pm

from bn_testing.terms import (
    Term,
    Linear,
    Polynomial,
)


class TestCompositions(unittest.TestCase):

    def setUp(self):
        self.parents = ['x', 'y']
        self.t_a = Linear(self.parents, coefs=[1, 2])
        self.t_b = Linear(self.parents, coefs=[-1, -2])
        self.mapping = {'x': pm.math.constant(1), 'y': pm.math.constant(2)}

    def test_addition(self):
        a = self.t_a + self.t_b
        result = a.apply(self.mapping)
        self.assertEqual(result.eval(), 0)

    def test_multiplication_with_coefs(self):
        a = 3*self.t_a*self.t_b
        result = a.apply(self.mapping)
        self.assertEqual(result.eval(), 3*(1*1+2*2)*(-1*1-2*2))

    def test_left_right_multiplication(self):
        left = 4*self.t_a
        right = self.t_a*4
        self.assertEqual(
            left.apply(self.mapping).eval(),
            right.apply(self.mapping).eval(),
        )

    def test_powers(self):
        result = self.t_b**3
        self.assertEqual(
            result.apply(self.mapping).eval(),
            (-1*1-2*2)**3
        )


class TestPolynomial(unittest.TestCase):

    def test_eval(self):
        polynomial = Polynomial(
            parents=['x', 'y'],
            exponents=[[1, 2]],
            coefs=[2.5],
            intercept=7,
        )

        self.assertEqual(
            polynomial.apply(
                {'x': pm.math.constant(1), 'y': pm.math.constant(2)}
            ).eval(),
            2.5*1**1*2**2+7
        )

    def test_negative_exponents(self):
        polynomial = Polynomial(
            parents=['x', 'y'],
            exponents=[[1, -2]],
            coefs=[2.5],
        )
        self.assertAlmostEqual(
            polynomial.apply(
                {'x': pm.math.constant(1.0), 'y': pm.math.constant(3.0)}
            ).eval(),
            2.5*1**1*3**(-2)
        )

    def test_value_error_on_missing_coef(self):
        with self.assertRaises(ValueError):
            Polynomial(
                parents=['x', 'y'],
                exponents=[[1, 2], [7, 8]],
                coefs=[1],
                intercept=7,
            )

    def test_value_error_on_wrong_exponent(self):
        with self.assertRaises(ValueError):
            Polynomial(
                parents=['x', 'y', 'z'],
                exponents=[[1, 2], [7, 8]],
                coefs=[1, 1],
                intercept=7,
            )

    def test_disp_under_parents_change(self):

        polynomial = Polynomial(
            parents=['x', 'y'],
            exponents=[[1, -2]],
            coefs=[2.5],
        )

        polynomial.parents = ['a', 'b']
        self.assertEqual(
            polynomial.disp,
            '2.5*a^1*b^-2'
        )

    def test_with_log(self):
        polynomial = Polynomial(
            parents=['x', 'y'],
            exponents=[[1, -2]],
            coefs=[2.5],
            with_log=True,
        )

        self.assertAlmostEqual(
            polynomial.apply(
                {'x': pm.math.constant(1.0), 'y': pm.math.constant(3.0)}
            ).eval(),
            2.5*np.log(1+np.abs(1**1*3**(-2)))
        )


class TestTermBasics(unittest.TestCase):

    def setUp(self):
        self.term = Term(
            parents=['a', 'b'],
            term_fn=lambda v: v['a']*v['b']
        )

    def test_call(self):
        result = self.term(
            a=pm.math.constant(10),
            b=pm.math.constant(1/10),
        )
        self.assertEqual(result.eval(), 1.0)

    def test_apply(self):

        result = self.term.apply(
            {
                'a': pm.math.constant(10),
                'b': pm.math.constant(1/10)
            }
        )
        self.assertEqual(result.eval(), 1.0)
