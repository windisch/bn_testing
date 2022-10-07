import networkx as nx
import numpy as np

from abc import ABCMeta
from itertools import combinations

from bn_testing.helpers import _generate_int_suffixes


class DAG(metaclass=ABCMeta):

    def generate(self, n_nodes, random=None):
        raise NotImplementedError()

    def init(self, random):
        self.random = random

    def make_node_names(self, n_nodes):
        return _generate_int_suffixes(
            prefix='f_',
            n=n_nodes)


class ScaleFree(DAG):

    def __init__(self, alpha=0.4, beta=0.5, gamma=0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def generate(self, n_nodes):

        dag = nx.scale_free_graph(
            n=n_nodes,
            seed=self.random,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            create_using=nx.DiGraph,
        )

        dag = nx.relabel_nodes(
            dag,
            mapping=dict(zip(dag.nodes(), self.make_node_names(n_nodes)))
        )
        return dag


class ErdosReny(DAG):

    def __init__(self, p=0.1):
        self.p = p

    def _select_edges(self, edges_iter, p):
        """
        TODO
        """

        # TODO Use np.fromiter
        edges = np.array([a for a in edges_iter])

        selection = self.random.choice(
            [False, True],
            p=[1-self.p, self.p],
            size=edges.shape[0]
        )
        return edges[selection]

    def generate(self, n_nodes):

        nodes = self.make_node_names(n_nodes)

        dag = nx.DiGraph()
        dag.add_nodes_from(nodes)

        # Shuffle nodes inplace
        self.random.shuffle(nodes)

        all_forward_edges = combinations(nodes, 2)
        edges_selected = self._select_edges(all_forward_edges, self.p)

        dag.add_edges_from(edges_selected)
        return dag
