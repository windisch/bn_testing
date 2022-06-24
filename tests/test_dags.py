import unittest
import networkx as nx

from bn_testing.graphs import GraphicalModel


class TestRandomDAG(unittest.TestCase):

    def setUp(self):
        self.bn = GraphicalModel(n_nodes=10, p=0.1)

    def test_acyclicity(self):
        self.assertTrue(nx.is_directed_acyclic_graph(self.bn.dag))

    def test_models(self):
        self.assertIsInstance(self.bn.models, dict)

    def test_sampling(self):
        df = self.bn.sample(100)
        self.assertTupleEqual(df.shape, (100, 10))
        self.assertSetEqual(set(df.columns), set(self.bn.nodes))
