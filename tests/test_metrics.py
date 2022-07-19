import unittest
import networkx as nx
import numpy as np

from bn_testing.metrics import (
    compute_group_distance_matrix,
    _group_iterator,
)
from bn_testing.dags import GroupedGaussianBN



def toy_distance_fn(adj_truth, adj_pred):
    return np.mean(np.abs((adj_truth-adj_pred)))

class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.model = GroupedGaussianBN(20, n_groups=4, p=0.2, random_state=10)
        self.dag = GroupedGaussianBN(20, n_groups=4, p=0.2, random_state=11)._generate_dag()

    def test_upper_triangular_matrix(self):

        distances = compute_group_distance_matrix(
            model=self.model,
            dag_learned=self.dag,
            distance_fn=toy_distance_fn
        )

        np.testing.assert_array_equal(
            distances - np.triu(distances),
            np.zeros((4, 4))
        )

    def test_zero_distance(self):
        distances = compute_group_distance_matrix(
            model=self.model,
            dag_learned=self.model.dag,
            distance_fn=toy_distance_fn
        )

        np.testing.assert_array_equal(
            distances,
            np.zeros((4, 4))
        )


class TestGroupIterator(unittest.TestCase):

    def test_ordering(self):
        self.assertListEqual(
            list(_group_iterator(['A', 'B', 'C'])),
            [
                ('A', 'A'),
                ('A', 'B'),
                ('A', 'C'),
                ('B', 'B'),
                ('B', 'C'),
                ('C', 'C'),
            ]
        )


