import numpy as np
import pandas as pd
import logging
import networkx as nx
from bn_testing.models import ConditionalGaussian
from itertools import product


logger = logging.getLogger(__name__)


class GraphicalModel(object):
    """

    Args:
        n_nodes (int): Number of nodes
        p (float): Erdös-Renyi probability
    """

    def __init__(self, n_nodes, p, random_state=10):

        self.p = p

        self.random_state = random_state
        self.random = np.random.RandomState(random_state)

        self.nodes = self._generate_nodes(n_nodes)

        logger.info('Generate DAG')
        self.dag = self._generate_dag()
        logger.info('Generate models')
        self.models = self._generate_models()

    def _generate_nodes(self, n_nodes):

        return [
            "f{}".format(
                str(i).zfill(len(str(n_nodes)))) for i in range(n_nodes)
        ]

    def _select_edges(self, edges, p):

        edges = np.array(edges)
        selection = self.random.choice([False, True], p=[1-self.p, self.p], size=len(edges))
        return edges[selection]

    def _generate_dag(self):
        all_edges = list(product(self.nodes, self.nodes))

        edges = self._select_edges(all_edges, self.p)
        self.random.shuffle(edges)

        dag = nx.DiGraph()
        dag.add_nodes_from(self.nodes)

        for node_from, node_to in edges:
            if node_from == node_to:
                continue

            if not nx.has_path(dag, node_to, node_from):
                dag.add_edge(node_from, node_to)

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

        nodes = nx.topological_sort(self.dag)

        for node in nodes:
            if node in self.models:
                df[node] = self.models[node].sample(df)
            else:
                # TODO: Change parameters
                df[node] = self.random.normal(loc=0, scale=1, size=n)
        return df

    def save(self):
        raise NotImplementedError()
