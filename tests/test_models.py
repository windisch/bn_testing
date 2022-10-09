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
                'f00', 'f01', 'f02', 'f03', 'f04',
                'f05', 'f06', 'f07', 'f08', 'f09',
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

    def test_order_of_transformations(self):
        nodes = list(self.model.transformations.keys())
        nodes_orig = nodes.copy()
        nodes.sort()
        self.assertListEqual(nodes, nodes_orig)


class TestModifications(unittest.TestCase):

    def setUp(self):
        self.model = BayesianNetwork(
            n_nodes=10,
            dag=ErdosReny(p=0.1),
            conditionals=LinearConditional(),
            random_state=10
        )

    def test_modification(self):
        transform_orig = self.model.transformations['f07']

        self.model.modify_transformation('f07')
        transform_new = self.model.transformations['f07']
        self.assertTrue(
            np.all(transform_new.coefs != transform_orig.coefs)
        )


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
