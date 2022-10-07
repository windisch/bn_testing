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
        self.variables = self._generate_variables()

    @property
    def nodes(self):
        return list(self.dag.nodes())

    def _generate_variables(self):
        variables = {}

        for node in nx.topological_sort(self.dag):
            parents = [variables[node] for node, _ in self.dag.in_edges(node)]

            if len(parents) > 0:
                variables[node] = sigmoid(self.conditionals(parents))
            else:
                # TODO
                variables[node] = pm.Beta.dist(
                    alpha=self.random.uniform(1, 5),
                    beta=self.random.uniform(1, 5),
                )

        return variables

    def sample(self, n):
        """
        """
        data = pm.draw([self.variables[n] for n in self.nodes], draws=n)
        df = pd.DataFrame(
            data=np.array(data).T,
            columns=self.nodes
        )
        return df
