import networkx as nx
import numpy as np

from abc import ABCMeta
from itertools import combinations

from bn_testing.helpers import _generate_int_suffixes


class DAG(metaclass=ABCMeta):

    def generate(self):
        raise NotImplementedError()

    def init(self, random):
        self.random = random

    def mark_as_hidden(self, dag, nodes):
        """
        TODO
        """
        for node in nodes:
            dag.nodes[node]['is_hidden'] = True
        return dag

    @staticmethod
    def show(dag):
        pos = nx.spring_layout(dag, seed=0)
        nx.draw_networkx_nodes(dag, pos=pos, node_size=100)
        nx.draw_networkx_edges(dag, pos=pos)
        nx.draw_networkx_labels(dag, pos=pos, font_size=6)


class RandomizedDAG(DAG):

    def __init__(self, n_visible_nodes, n_hidden_nodes=0):
        self.n_visible_nodes = n_visible_nodes
        self.n_hidden_nodes = n_hidden_nodes

        self.nodes_visible = _generate_int_suffixes(
            prefix='X',
            n=self.n_visible_nodes)

        self.nodes_hidden = _generate_int_suffixes(
            prefix='H',
            n=self.n_hidden_nodes)

    @property
    def nodes(self):
        return self.nodes_hidden + self.nodes_visible

    @property
    def n_nodes(self):
        return self.n_hidden_nodes + self.n_visible_nodes


class ScaleFree(RandomizedDAG):

    def __init__(self, n_visible_nodes=None, alpha=0.4, beta=0.5, gamma=0.1, n_hidden_nodes=0):
        RandomizedDAG.__init__(
            self,
            n_visible_nodes=n_visible_nodes,
            n_hidden_nodes=n_hidden_nodes)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def generate(self):

        dag = nx.scale_free_graph(
            n=self.n_nodes,
            seed=self.random,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            create_using=nx.DiGraph,
        )

        dag = nx.relabel_nodes(
            dag,
            mapping=dict(zip(dag.nodes(), self.nodes))
        )

        dag = self.mark_as_hidden(dag, self.nodes_hidden)
        return dag


class ErdosReny(RandomizedDAG):
    """

    :param float p: Erdös-Renyi probability
    :param int n_visible_nodes: Number of visible nodes
    :param int n_hidden_nodes: Number of hidden nodes
    """

    def __init__(self, n_visible_nodes=None, p=0.1, n_hidden_nodes=0):
        RandomizedDAG.__init__(
            self,
            n_visible_nodes=n_visible_nodes,
            n_hidden_nodes=n_hidden_nodes)
        self.p = p

    def _select_edges(self, edges_iter, p):
        """
        Creates a randomized sublist of :code:`edges_iter` where each edge is in that sublist with
        probability :code:`p`

        Args:
            edge_iter (iterable): Iterable that yields edges
            p (float): Selection probability

        Returns:
            list: Sublist of :code:`edges_iter`
        """

        edges = np.array([a for a in edges_iter])

        selection = self.random.choice(
            [False, True],
            p=[1-self.p, self.p],
            size=edges.shape[0]
        )
        return edges[selection]

    def generate(self):

        dag = nx.DiGraph()
        dag.add_nodes_from(self.nodes)

        # Shuffle nodes inplace
        self.random.shuffle(self.nodes)

        all_forward_edges = combinations(self.nodes, 2)
        edges_selected = self._select_edges(all_forward_edges, self.p)

        dag.add_edges_from(edges_selected)
        dag = self.mark_as_hidden(dag, self.nodes_hidden)
        return dag
