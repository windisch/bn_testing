import numpy as np
import pandas as pd
import logging
import networkx as nx
import pymc as pm

from abc import ABCMeta

from bn_testing.helpers import sigmoid

logger = logging.getLogger(__name__)


class BayesianNetwork(metaclass=ABCMeta):
    """

    Note:
        We assume that the order of the nodes is a topological ordering of the resulting graph.

    Args:
        n_nodes (int): Number of nodes
        p (float): Erdös-Renyi probability
    """

    def __init__(self, dag=None, conditionals=None, n_nodes=None, random_state=None):
        self.random_state = random_state
        self.random = np.random.RandomState(self.random_state)

        self.dag_gen = dag
        self.dag_gen.init(self.random)

        self.conditionals = conditionals
        self.conditionals.init(self.random)

        self.n_nodes = n_nodes

        self.generate()

    def generate(self):
        """
        Generates the mod
        """
        logger.info('Generate DAG')
        self.dag = self.dag_gen.generate(self.n_nodes)
        logger.info('Generate models')
        self.transformations = self._build_transformations()

    @property
    def nodes(self):
        return list(self.dag.nodes())

    def _build_transformations(self):
        transformations = {}
        for node in nx.topological_sort(self.dag):
            n_parents = self.dag.in_degree(node)
            if n_parents > 0:
                transformations[node] = self.conditionals.make_transformation(
                    n_parents=n_parents,
                )
        return transformations

    def _build_variable(self, node, parents):
        if len(parents) > 0:
            var = sigmoid(
                self.transformations[node].apply(parents) + self.conditionals.make_noise()
            )
        else:
            var = self.conditionals.make_source()
        return var

    def _build_variables(self):
        variables = {}

        for node in nx.topological_sort(self.dag):
            parents = [variables[n] for n, _ in self.dag.in_edges(node)]
            variables[node] = self._build_variable(node, parents)
        return variables

    def sample(self, n):
        """
        """

        logger.info('Build variables')
        variables = self._build_variables()

        logger.info('Start sampling')
        data = pm.draw([variables[n] for n in self.nodes], draws=n)
        df = pd.DataFrame(
            data=np.array(data).T,
            columns=self.nodes
        )
        return df
