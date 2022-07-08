import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import networkx as nx
from bn_testing.models import ConditionalGaussian
from itertools import (
    combinations,
    product,
)


logger = logging.getLogger(__name__)


def _generate_int_suffixes(prefix, n):
    return [
        "{}{}".format(
            prefix,
            str(i).zfill(len(str(n)))) for i in range(n)
    ]


class GraphicalModel(object):
    """


    Note:
        We assume that the order of the nodes is the topological ordering of the resulting graph

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

    def _generate_dag(self):

        dag = nx.DiGraph()
        dag.add_nodes_from(self.nodes)

        all_edges = []

        # Select random edges within the groups
        for group, nodes in self.groups.items():

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
