import unittest
import networkx as nx
import numpy as np

from bn_testing.models import BayesianNetwork
from bn_testing.dags import ErdosReny

from bn_testing.transformations import Linear
from bn_testing.conditionals import (
    LinearConditional,
    PolynomialConditional,
)


class TestLinearErdosReny(unittest.TestCase):

    def setUp(self):
        self.model = BayesianNetwork(
            n_nodes=10,
            dag=ErdosReny(p=0.1),
            conditionals=LinearConditional(),
            random_state=10
        )

    def test_acyclicity(self):
        self.assertTrue(nx.is_directed_acyclic_graph(self.model.dag))

    def test_nodes(self):
        np.testing.assert_array_equal(
            self.model.nodes,
            [
                'f_00', 'f_01', 'f_02', 'f_03', 'f_04',
                'f_05', 'f_06', 'f_07', 'f_08', 'f_09',
            ]
        )

    def test_transformations(self):
        self.assertIsInstance(self.model.transformations, dict)
        for t in self.model.transformations.values():
            self.assertIsInstance(t, Linear)

    def test_sampling(self):
        df = self.model.sample(100)
        self.assertTupleEqual(df.shape, (100, 10))
        self.assertSetEqual(set(df.columns), set(self.model.nodes))


class TestPolynomialErdosReny(unittest.TestCase):

    def setUp(self):
        self.model = BayesianNetwork(
            n_nodes=10,
            dag=ErdosReny(p=0.1),
            conditionals=PolynomialConditional(),
            random_state=10
        )

    def test_sampling(self):
        df = self.model.sample(100)
        self.assertTupleEqual(df.shape, (100, 10))
        self.assertSetEqual(set(df.columns), set(self.model.nodes))
