import unittest
import networkx as nx
import numpy as np

from bn_testing.dags import GroupedGaussianBN


class TestRandomDAG(unittest.TestCase):

    def setUp(self):
        self.model = GroupedGaussianBN(n_nodes=10, n_groups=2, p=0.5)

    def test_acyclicity(self):
        self.assertTrue(nx.is_directed_acyclic_graph(self.model.dag))

    def test_nodes(self):
        np.testing.assert_array_equal(
            self.model.nodes,
            [
                'g0_f0', 'g0_f1', 'g0_f2', 'g0_f3', 'g0_f4',
                'g1_f0', 'g1_f1', 'g1_f2', 'g1_f3', 'g1_f4',
            ]
        )

    def test_models(self):
        self.assertIsInstance(self.model.models, dict)

    def test_sampling(self):
        df = self.model.sample(100)
        self.assertTupleEqual(df.shape, (100, 10))
        self.assertSetEqual(set(df.columns), set(self.model.nodes))

    def test_get_nodes_of_group(self):
        self.assertListEqual(
            self.model._get_nodes_of_group('g0'),
            ['g0_f0', 'g0_f1', 'g0_f2', 'g0_f3', 'g0_f4'],
        )

    def test_get_nodes_of_group_exception_on_unkown_group(self):
        with self.assertRaises(ValueError):
            self.model._get_nodes_of_group('g10')

    def test_get_grouped_subgraph(self):
        dag = self.model.get_subgraph_on_groups(['g0'])

        for edge in dag.edges:
            for node in edge:
                self.assertTrue(node.startswith('g0'))

    def test_structural_zeros(self):
        structural_zeros = self.model.get_structural_zeros()
        self.assertEqual(len(structural_zeros), 10/2*10/2)
        self.assertIn(('g1_f0', 'g0_f1'), structural_zeros)
        self.assertIn(('g1_f1', 'g0_f1'), structural_zeros)
        self.assertIn(('g1_f1', 'g0_f0'), structural_zeros)

    def test_check_varsortability(self):
        df = self.model.sample(100)
        np.testing.assert_array_almost_equal(df.std(), 1.0)
        np.testing.assert_array_almost_equal(df.mean(), 0.0)
