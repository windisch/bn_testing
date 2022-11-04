import unittest
from unittest.mock import patch
import networkx as nx
import numpy as np
import pandas as pd
import os
import tempfile
import pymc as pm

from scipy.stats import ks_2samp

from bn_testing.models import BayesianNetwork
from bn_testing.dags import (
    ErdosReny,
    DAG,
)

from bn_testing.terms import Linear
from bn_testing.conditionals import (
    LinearConditional,
    PolynomialConditional,
    ConstantConditional,
)


class ToyDAG(DAG):

    def make_dag(self):
        dag = nx.DiGraph()
        dag.add_edges_from([['A', 'B'], ['B', 'D'], ['C', 'D'], ['D', 'E']])
        return dag


class ToyDAGWithTerms(DAG):

    def make_dag(self):
        dag = nx.DiGraph()
        dag.add_edges_from([['A', 'B'], ['B', 'D'], ['C', 'D'], ['D', 'E']])

        dag.nodes['D']['term'] = Linear(['B', 'C'], [-100, 100])
        dag.nodes['D']['noise'] = pm.math.constant(0)
        return dag


class ToyDAGWithoutNoise(DAG):

    def make_dag(self):
        dag = nx.DiGraph()

        dag.add_node(
            'C',
            term=Linear(['A', 'B'], [1, 2]),
            no_noise=True,
        )

        dag.add_edges_from([['A', 'C'], ['B', 'C']])
        return dag


class ToyDAGWithHiddenNode(DAG):

    def make_dag(self):
        dag = nx.DiGraph()
        dag.add_edges_from([['A', 'B'], ['B', 'D'], ['C', 'D'], ['D', 'E']])

        dag.nodes['D']['is_hidden'] = True
        return dag


class ToyDAGWithFixedSource(DAG):

    def make_dag(self):
        dag = nx.DiGraph()
        dag.add_edges_from([['A', 'B'], ['B', 'C'], ['D', 'C']])
        dag.nodes['A']['distribution'] = pm.Normal.dist(mu=10, sigma=5)
        return dag


class TestLinearErdosReny(unittest.TestCase):

    def setUp(self):
        self.model = BayesianNetwork(
            dag=ErdosReny(p=0.1, n_visible_nodes=10, n_hidden_nodes=2),
            conditionals=LinearConditional(),
            random_state=10,
        )

    def test_acyclicity(self):
        self.assertTrue(nx.is_directed_acyclic_graph(self.model.dag))

    def test_visible_nodes(self):
        np.testing.assert_array_equal(
            self.model.visible_nodes,
            [
                'X00', 'X01', 'X02', 'X03', 'X04',
                'X05', 'X06', 'X07', 'X08', 'X09',
            ]
        )

    def test_hidden_nodes(self):
        np.testing.assert_array_equal(
            self.model.hidden_nodes,
            [
                'H0', 'H1'
            ]
        )

    def test_nodes(self):
        np.testing.assert_array_equal(
            self.model.nodes,
            [
                'H0', 'H1',
                'X00', 'X01', 'X02', 'X03', 'X04',
                'X05', 'X06', 'X07', 'X08', 'X09',
            ]
        )

    def test_terms(self):
        self.assertIsInstance(self.model.terms, dict)
        for t in self.model.terms.values():
            self.assertIsInstance(t, Linear)

    def test_sampling(self):
        df = self.model.sample(100)
        self.assertTupleEqual(df.shape, (100, 10))
        self.assertSetEqual(set(df.columns), set(self.model.visible_nodes))

    def test_sampling_with_hidden(self):
        df = self.model.sample(100, exclude_hidden_nodes=False)

        for node in self.model.dag_gen.nodes_hidden:
            self.assertIn(node, df)
        self.assertTupleEqual(df.shape, (100, 12))

    def test_sampling_of_some_variables(self):
        df = self.model.sample(100, nodes=['X00', 'X09'])
        self.assertTupleEqual(df.shape, (100, 2))
        self.assertListEqual(
            df.columns.to_list(),
            ['X00', 'X09'],
        )

    def test_order_of_terms(self):
        nodes = list(self.model.terms.keys())
        nodes_orig = nodes.copy()
        nodes.sort()
        self.assertListEqual(nodes, nodes_orig)

    def test_normalized_sampling(self):
        df = self.model.sample(100, normalize=True)
        np.testing.assert_array_almost_equal(df.std(), 1.0)

    def test_set_term_with_wrong_parents(self):

        with self.assertRaises(AssertionError):
            self.model._set_term(
                node='X03',
                term=Linear(['Y1', 'Y2'], [0.1, 0.2]),
            )


class TestFixedTerms(unittest.TestCase):

    def test_fixed_term(self):
        model = BayesianNetwork(
            dag=ToyDAGWithTerms(),
            conditionals=PolynomialConditional(),
            random_state=10
        )
        fixed_term = model.terms['D']
        self.assertListEqual(fixed_term. coefs, [-100, 100])
        self.assertIsInstance(fixed_term, Linear)

    def test_no_noise(self):
        model = BayesianNetwork(
            dag=ToyDAGWithoutNoise(),
            conditionals=LinearConditional(),
            random_state=10
        )

        df = model.sample(10)
        np.testing.assert_array_equal(
            df['C'],
            df['A'] + 2*df['B']
        )


class TestModelWithHiddenNOdes(unittest.TestCase):

    def setUp(self):
        self.model = BayesianNetwork(
            dag=ToyDAGWithHiddenNode(),
            conditionals=PolynomialConditional(),
            random_state=10
        )

    def test_sampling(self):
        df = self.model.sample(10)
        self.assertNotIn('D', df)
        self.assertTupleEqual(df.shape, (10, 4))


class TestModifications(unittest.TestCase):

    def setUp(self):
        self.model = BayesianNetwork(
            dag=ToyDAG(),
            conditionals=LinearConditional(),
            random_state=10
        )

    def test_modification_of_source_node(self):

        # Copy original distributions
        dists_orig = dict(self.model.sources)

        self.model.modify_source_node(
            node='A',
            distribution=pm.Normal.dist(mu=0),
        )

        for node in self.model.nodes:
            if not self.model.is_source(node):
                continue

            if node == 'A':
                self.assertNotEqual(
                    dists_orig[node],
                    self.model.sources[node],
                )
            else:
                self.assertEqual(
                    dists_orig[node],
                    self.model.sources[node],
                )

    def test_assert_that_modify_source_is_called(self):
        with patch.object(self.model, 'modify_source_node') as mock:
            self.model.modify_node('A')
            mock.assert_called_once()

    def test_assert_that_modify_inner_is_called(self):
        with patch.object(self.model, 'modify_inner_node') as mock:
            self.model.modify_node('B')
            mock.assert_called_once()

    def test_modify_source_node_for_non_source_node(self):
        with self.assertRaises(ValueError):
            self.model.modify_source_node(
                node='B',
                conditionals=LinearConditional(),
            )

    def test_modify_inner_node_for_source_node(self):
        with self.assertRaises(ValueError):
            self.model.modify_inner_node(
                node='A',
                conditionals=LinearConditional(),
            )

    def test_setting_of_concreate_term_with_wrong_parents(self):
        with self.assertRaises(AssertionError):
            self.model.modify_inner_node(
                node='B',
                term=Linear(['D', 'E'], [0.1, 0.2]),
            )

    def test_modify_node_only_with_node_name(self):
        term_orig = self.model.terms['D']

        self.model.modify_node('D')
        term_new = self.model.terms['D']
        self.assertTrue(
            np.all(term_new.coefs != term_orig.coefs)
        )


class TestPolynomialErdosReny(unittest.TestCase):

    def setUp(self):
        self.model = BayesianNetwork(
            dag=ErdosReny(p=0.1, n_visible_nodes=10),
            conditionals=PolynomialConditional(),
            random_state=10
        )

    def test_sampling(self):
        df = self.model.sample(100)
        self.assertTupleEqual(df.shape, (100, 10))
        self.assertSetEqual(set(df.columns), set(self.model.nodes))


class TestCausalEffects(unittest.TestCase):

    def setUp(self):
        self.model = BayesianNetwork(
            dag=ToyDAG(),
            conditionals=LinearConditional(),
            random_state=10
        )

    def test_computation_of_causal_effect(self):
        effect = self.model.compute_average_causal_effect(
            node_from='B',
            node_onto='E',
            value=3)

        self.assertGreater(effect, 1)

    def test_assure_clean_up_after_computation(self):
        self.model.compute_average_causal_effect(
            node_from='B',
            node_onto='E',
            value=1)
        self.assertTrue(not isinstance(self.model.terms['B'], ConstantConditional))
        self.assertTrue(self.model.noises['B'] != pm.math.constant(0))
        self.assertTrue(isinstance(self.model.noises['B'], type(LinearConditional().make_noise())))

    def test_average_causal_effect_on_source_node(self):
        effect = self.model.compute_average_causal_effect(
            node_from='A',
            node_onto='E',
            value=2)
        self.assertGreater(effect, 0.4)


class TestVarsortability(unittest.TestCase):

    def setUp(self):
        self.model = BayesianNetwork(
            dag=ToyDAG(),
            conditionals=LinearConditional(),
            random_state=10
        )

    def test_compute_varsortability(self):
        varsortability = self.model.compute_varsortability()
        self.assertLessEqual(varsortability, 1)
        self.assertGreaterEqual(varsortability, 0)


class TestSavingAndLoading(unittest.TestCase):

    def test_saving_and_loading(self):
        model = BayesianNetwork(
            dag=ToyDAGWithTerms(),
            conditionals=LinearConditional(),
            random_state=10
        )

        with tempfile.NamedTemporaryFile() as tmpfile:
            model.save(tmpfile.name)
            self.assertTrue(os.path.exists(tmpfile.name))
            model_loaded = BayesianNetwork.load(tmpfile.name)

        pd.testing.assert_frame_equal(
            model.sample(100),
            model_loaded.sample(100),
        )


class TestFixedSourceDistributions(unittest.TestCase):

    def setUp(self):

        self.model = BayesianNetwork(
            dag=ToyDAGWithFixedSource(),
            conditionals=LinearConditional(),
            random_state=10
        )

    def test_source_distribution_object(self):
        self.assertEqual(
            self.model.sources['A'],
            self.model.dag.nodes['A']['distribution'],
        )

    def test_different_means(self):
        df = self.model.sample(1000)

        _, p = ks_2samp(df['A'], df['D'])

        self.assertLess(p, 0.001)
        self.assertGreater(df['A'].mean(), 9)
        self.assertLess(df['D'].mean(), 1)
