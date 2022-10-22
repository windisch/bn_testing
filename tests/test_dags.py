import unittest
import numpy as np
import networkx as nx

from bn_testing.dags import (
    ErdosReny,
    DAG,
)


class ToyGraphWithCycle(DAG):

    def make_dag(self):
        return nx.cycle_graph(5, create_using=nx.DiGraph)


class TestDAGs(unittest.TestCase):

    def test_exception_on_cycles(self):
        with self.assertRaises(AssertionError):
            ToyGraphWithCycle().generate()


class TestErdosReny(unittest.TestCase):

    def setUp(self):
        self.random = np.random.RandomState(1)
        self.dag_gen = ErdosReny(n_visible_nodes=10, n_hidden_nodes=5)
        self.dag = self.dag_gen.generate(self.random)

    def test_number_of_hidden_nodes(self):
        self.assertEqual(self.dag.number_of_nodes(), 10 + 5)
        self.assertListEqual(
            self.dag_gen.nodes_hidden,
            ['H0', 'H1', 'H2', 'H3', 'H4']
        )

    def test_attribute_of_hidden_nodes(self):
        for node in self.dag_gen.nodes_hidden:
            self.assertTrue(self.dag.nodes[node]['is_hidden'])

    def test_attribute_of_visible_nodes(self):

        for node in self.dag_gen.nodes_visible:
            self.assertNotIn('is_hidden', self.dag.nodes[node])
