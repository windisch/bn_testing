import unittest
import networkx as nx
import numpy as np

from bn_testing.dags import GroupedGaussianBN


class TestRandomDAG(unittest.TestCase):

    def setUp(self):
        self.bn = GroupedGaussianBN(n_nodes=10, n_groups=2, p=0.1)

    def test_acyclicity(self):
        self.assertTrue(nx.is_directed_acyclic_graph(self.bn.dag))

    def test_group_names(self):
        assert False

    def test_nodes(self):
        np.testing.assert_array_equal(
            self.bn.nodes,
            [
                'g0_f0', 'g0_f1', 'g0_f2', 'g0_f3', 'g0_f4',
                'g1_f0', 'g1_f1', 'g1_f2', 'g1_f3', 'g1_f4',
            ]
        )

    def test_models(self):
        self.assertIsInstance(self.bn.models, dict)

    def test_sampling(self):
        df = self.bn.sample(100)
        self.assertTupleEqual(df.shape, (100, 10))
        self.assertSetEqual(set(df.columns), set(self.bn.nodes))
