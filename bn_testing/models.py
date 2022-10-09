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
        dag (bn_testing.dags.DAG): A DAG generation method
        conditionals (bn_testing.conditionals.Conditionals): A conditional type
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
    def edges(self):
        return list(self.dag.edges())

    def modify_transformation(self, node, conditionals=None):

        if node not in self.nodes:
            raise ValueError(f'Unkown node {node}')

        if conditionals is None:
            conditionals = self.conditionals
        else:
            conditionals.init(self.random)
        self.transformations[node] = conditionals(self._get_parents(node))

    @property
    def nodes(self):
        return list(self.dag.nodes())

    def _get_parents(self, node):
        return [n for n, _ in self.dag.in_edges(node)]

    def _build_transformations(self):
        transformations = {}
        for node in self.nodes:
            parents = self._get_parents(node)
            if len(parents) > 0:
                transformations[node] = self.conditionals.make_transformation(
                    parents=parents,
                )
        return transformations

    def _build_variable(self, node, parents_mapping):
        """
        Builds the variable corresponding to `node` by invoking the transformation created by the
        conditionals.

        Args:
            node (str): Name of the nodes
            parents_mapping (dict): Mapping of node names to random variables.

        """
        if len(parents_mapping.keys()) > 0:
            var = sigmoid(
                self.transformations[node].apply(parents_mapping) + self.conditionals.make_noise()
            )
        else:
            var = self.conditionals.make_source()
        return var

    def _build_variables(self):
        variables = {}

        for node in nx.topological_sort(self.dag):
            parents_mapping = {
                parent: variables[parent] for parent, _ in self.dag.in_edges(node)
            }
            variables[node] = self._build_variable(node, parents_mapping)
        return variables

    def sample(self, n):
        """
        Samples `n` many identic and independent observations from the Bayesian network.


        Args:
            n (int): Number of observation to be created

        Returns:
            pandas.DataFrame: Dataframe in which the variables are columns and the observations are
            rows
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

    def show(self):
        """
        Visualizes the generated DAG.
        """
        self.dag_gen.show(self.dag)
