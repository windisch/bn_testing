import numpy as np
import pandas as pd
import logging
import networkx as nx
import pymc as pm

from abc import ABCMeta

from bn_testing.conditionals import ConstantConditional

logger = logging.getLogger(__name__)


class BayesianNetwork(metaclass=ABCMeta):
    """

    Note:
        We assume that the order of the nodes is a topological ordering of the resulting graph.

    Args:
        p (float): Erdös-Renyi probability
        dag (bn_testing.dags.DAG): A DAG generation method
        conditionals (bn_testing.conditionals.Conditionals): A conditional type
    """

    def __init__(self, dag=None, conditionals=None, random_state=None):
        self.random_state = random_state
        self.random = np.random.RandomState(self.random_state)

        self.dag_gen = dag
        self.dag_gen.init(self.random)

        self.conditionals = conditionals
        self.conditionals.init(self.random)

        self.generate()
        self.n_nodes = self.dag.number_of_nodes

    def generate(self):
        """
        Generates the mod
        """
        logger.info('Generate DAG')
        self.dag = self.dag_gen.generate()
        logger.info('Generate models')
        self.transformations = self._build_transformations()
        self.sources = self._build_sources()
        self.noises = self._build_noises()

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
        self.transformations[node] = conditionals.make_transformation(self._get_parents(node))
        self.noises[node] = conditionals.make_noise()

    @property
    def nodes(self):
        return list(self.dag.nodes())

    def _get_parents(self, node):
        return [n for n, _ in self.dag.in_edges(node)]

    def is_source(self, node):
        """
        Checks if the given node is a source node, i.e., has no parents.

        Args:
            node (str): Name of a node in the dag

        Returns:
            bool: True if node is a source, false otherwise
        """
        return len(self._get_parents(node)) == 0

    def _build_noises(self):
        noises = {}
        for node in self.nodes:
            parents = self._get_parents(node)
            if len(parents) > 0:
                noises[node] = self.conditionals.make_noise()
        return noises

    def _build_sources(self):
        """
        Builds PyMC variables that correspond to source nodes in the DAG.
        """
        sources = {}
        for node in self.nodes:
            if self.dag.in_degree(node) == 0:
                sources[node] = self.conditionals.make_source()
        return sources

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
            var = self.transformations[node].apply(parents_mapping) + self.noises[node]
        else:
            # As generating the source introduces some randomness we would like to fix, we have to
            # build the source variables only once and reuse them here.
            var = self.sources[node]
        return var

    def compute_average_causal_effect(self, node_from=None, node_onto=None, value=None, n=1000):
        """
        TODO
        """

        if self.is_source(node_from):
            raise NotImplementedError()
        else:

            df_orig = self.sample(n=n, nodes=[node_onto])

            transformation = self.transformations[node_from]
            noise = self.noises[node_from]

            self.modify_transformation(
                node=node_from,
                conditionals=ConstantConditional(value=value)
            )

            df_intervent = self.sample(n=n, nodes=[node_onto])

            self.transformations[node_from] = transformation
            self.noises[node_from] = noise

            return df_intervent[node_onto].mean() - df_orig[node_onto].mean()

    def _build_variables(self):
        variables = {}

        for node in nx.topological_sort(self.dag):
            parents_mapping = {
                parent: variables[parent] for parent, _ in self.dag.in_edges(node)
            }

            variables[node] = self._build_variable(node, parents_mapping)
        return variables

    def sample(self, n, nodes=None):
        """
        Samples `n` many identic and independent observations from the Bayesian network.


        Args:
            n (int): Number of observation to be created

        Returns:
            pandas.DataFrame: Dataframe in which the variables are columns and the observations are
            rows
        """

        if nodes is None:
            nodes = self.nodes

        logger.info('Build variables')
        variables = self._build_variables()

        logger.info('Start sampling')
        data = pm.draw([variables[n] for n in nodes], draws=n)
        df = pd.DataFrame(
            data=np.array(data).T,
            columns=nodes
        )
        return df

    def show(self):
        """
        Visualizes the generated DAG.
        """
        self.dag_gen.show(self.dag)
