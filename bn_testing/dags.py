import networkx as nx
import numpy as np

from abc import ABCMeta
from itertools import combinations

from bn_testing.helpers import (
    _generate_int_suffixes,
    _make_random_state,
)


class DAG(metaclass=ABCMeta):

    def make_dag(self):
        raise NotImplementedError()

    def generate(self, random=None):
        """
        """
        random = _make_random_state(random)
        self.init(random)
        dag = self.make_dag()
        assert nx.is_directed_acyclic_graph(dag), 'Cycles detected'
        return dag

    def init(self, random):
        self.random = random

    def mark_as_hidden(self, dag, nodes):
        """
        Sets for the stated nodes in the dag the node attribute :code:`is_hidden` to `True`.

        :params networkx.DiGraph dag: The DAG whose nodes should be hidden
        :params list nodes: List of nodes in the DAG

        :returns: The same DAG where the stated nodes have the additional attribute
            :code:`is_hidden` set to True
        :rtype: networkx.DiGraph

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
    """
    Helper class to generate randomized DAGs.


    :param int n_visible_nodes: The number of visible nodes
    :param int n_hidden_nodes: The number of hidden nodes. Defaults to :code:`0`.
    """

    def __init__(self, n_visible_nodes, n_hidden_nodes=0):
        self.n_visible_nodes = n_visible_nodes
        self.n_hidden_nodes = n_hidden_nodes

        self.nodes_visible = _generate_int_suffixes(
            prefix='X',
            n=self.n_visible_nodes)

        self.nodes_hidden = _generate_int_suffixes(
            prefix='H',
            n=self.n_hidden_nodes)

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

    @property
    def nodes(self):
        return self.nodes_hidden + self.nodes_visible

    @property
    def n_nodes(self):
        return self.n_hidden_nodes + self.n_visible_nodes

    def generate(self, random):
        dag = super(RandomizedDAG, self).generate(random)
        dag = self.mark_as_hidden(dag, self.nodes_hidden)
        return dag


class ScaleFree(RandomizedDAG):
    """
    Generates a scale free DAG.
    """

    def __init__(self, n_visible_nodes=None, alpha=0.4, beta=0.5, gamma=0.1, n_hidden_nodes=0):
        RandomizedDAG.__init__(
            self,
            n_visible_nodes=n_visible_nodes,
            n_hidden_nodes=n_hidden_nodes)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def make_dag(self):

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
    A Directed a acyclic graph generated with the Erdos-Reny model.

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

    def make_dag(self):
        dag = nx.DiGraph()
        dag.add_nodes_from(self.nodes)

        # Shuffle nodes inplace
        self.random.shuffle(self.nodes)

        all_forward_edges = combinations(self.nodes, 2)
        edges_selected = self._select_edges(all_forward_edges, self.p)

        dag.add_edges_from(edges_selected)
        dag = self.mark_as_hidden(dag, self.nodes_hidden)
        return dag
