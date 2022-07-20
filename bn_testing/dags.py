import numpy as np
import pandas as pd
import logging
from itertools import chain
from tqdm import tqdm
import networkx as nx
from itertools import (
    combinations,
    product,
)

from bn_testing.conditionals import ConditionalGaussian
from bn_testing.helpers import _generate_int_suffixes

logger = logging.getLogger(__name__)


class GroupedGaussianBN(object):
    """

    Note:
        We assume that the order of the nodes is a topological ordering of the resulting graph.

    Args:
        n_nodes (int): Number of nodes
        p (float): Erdös-Renyi probability
    """

    def __init__(self, n_nodes, n_groups=1, p=0.01, random_state=10):

        self.p = p

        self.random_state = random_state
        self.random = np.random.RandomState(random_state)

        self.group_names = self._generate_group_names(n_groups)

        self.groups = {
            group_name: self._generate_node_names(
                # TODO: Improve splitting
                # How about specifying nodes_per_group and avoid
                # that size of the last group is uncontrolled
                n_nodes=int(n_nodes/n_groups),
                group_name=group_name,
            ) for group_name in self.group_names
        }

        logger.info('Generate DAG')
        self.dag = self._generate_dag()
        logger.info('Generate models')
        self.models = self._generate_models()

    @property
    def nodes(self):
        return np.concatenate(
            tuple([self.groups[g] for g in self.group_names])
        )

    def _generate_group_names(self, n_groups):
        return _generate_int_suffixes(
            prefix="g",
            n=n_groups,
        )

    def _generate_node_names(self, n_nodes, group_name):

        return _generate_int_suffixes(
            prefix=group_name + "_f",
            n=n_nodes
        )

    def _select_edges(self, edges_iter, p):

        # TODO Use np.fromiter
        edges = np.array([a for a in edges_iter])

        selection = self.random.choice(
            [False, True],
            p=[1-self.p, self.p],
            size=edges.shape[0]
        )
        return edges[selection]

    def _get_nodes_of_group(self, group_name):

        if group_name not in self.group_names:
            raise ValueError('Unkown group {group_name}')
        return [n for n in self.dag.nodes() if n.startswith(group_name)]

    def get_subgraph_on_groups(self, groups):
        all_nodes = [self._get_nodes_of_group(group) for group in groups]
        nodes = [n for nodes in all_nodes for n in nodes]
        return self.dag.subgraph(nodes)

    def _generate_dag(self):

        dag = nx.DiGraph()
        dag.add_nodes_from(self.nodes)

        all_edges = []

        # Select random edges within the groups
        for group, nodes in self.groups.items():

            # combinations will only give you (A,B) but but not (B,A), is that intended ?
            all_forward_edges_of_group = combinations(nodes, 2)
            edges_selected = self._select_edges(all_forward_edges_of_group, self.p)
            all_edges.extend(edges_selected)

        # Select edges within subsequent groups
        for group_from, group_to in combinations(self.group_names, 2):
            edges_within = product(
                self.groups[group_from],
                self.groups[group_to],
            )
            edges_selected = self._select_edges(edges_within, self.p)
            all_edges.extend(edges_selected)

        dag.add_edges_from(all_edges)
        return dag

    def get_structural_zeros(self):
        """
        Returns edges that are by definition not used by the model.
        """
        structural_zeros = []
        for group_to, group_from in combinations(self.group_names[::-1], 2):
            for edge_from, edge_to in product(self.groups[group_to], self.groups[group_from]):
                structural_zeros.append((edge_from, edge_to))
        return structural_zeros


    def _generate_models(self):
        models = {}
        for node in self.dag.nodes():
            parents = [node for node, _ in self.dag.in_edges(node)]

            if len(parents) > 0:
                models[node] = ConditionalGaussian(
                    parents=parents,
                    random_state=self.random_state)
        return models

    def sample(self, n):

        df = pd.DataFrame()

        for node in tqdm(self.nodes):
            if node in self.models:
                df[node] = self.models[node].sample(df)
            else:
                # TODO: Change parameters
                df[node] = self.random.normal(loc=0, scale=1, size=n)
        return df

    def save(self):
        raise NotImplementedError()
