Graph construction
==================


Available generators
--------------------


* :py:class:`~bn_testing.dags.ScaleFree`: Generates a scale-free dag
* :py:class:`~bn_testing.dags.ErdosReny`: Generates a DAG based on the
  Erdos-Reny model


Implementing own generators
---------------------------

To implement an own DAG generator, a subclass of
:py:class:`~bn_testing.dags.DAG` needs to be implemented where only
the :py:func:`~bn_testing.dags.DAG.make_dag` method needs to be
implemented. This method has to return an acyclic
:py:class:`networkx.DiGraph` object.

.. code-block:: python

   import networkx as nx
   from bn_testing.dags import DAG

   class PathGraph(DAG):

      def make_dag(self):
         return nx.path_graph(
            n=100,
            create_using=nx.DiGraph
         )


.. note::

   Acyclicity will be checked when the
   :py:class:`~bn_testing.models.BayesianNetwork` object receives the
   generated graph object from the generator. If the digraph contains
   directed cycles, an exception is thrown.



Randomization
-------------

The main usecase of :py:mod:`bn_testing` is to generate randomized
bayesian networks, where both, the graph and the conditionals are
randomly choosen. The class :py:class:`bn_testing.dags.RandomizedDAG`
brings utilities that ease the random generation. 

Here is an example to generate a subgraph of a path where 10% of the
edges are removed randomly:

.. code-block:: python

   import networkx as nx
   from bn_testing.dags import RandomizedDAG

   class RandomizedPathSubGraph(RandomizedDAG):

      def make_dag(self):
         # Generate a dag using self.n_nodes
         dag = nx.path_graph(
            n=self.n_nodes,
            create_using=nx.DiGraph
         )

         # Use self.random for any random selection
         edges_indices_to_remove = self.random.choice(
            a=np.arange(n_nodes-1), 
            size=int(0.1*n_nodes),
            replace=False)

         edges_to_remove = [
            e for i, e in enumerate(dag.edges()) if i in edges_indices_to_remove
         ]

         dag.remove_edges_from(edges_to_remove)
         return dag


In a model, this can be used as follows:

.. code-block:: python

   from bn_testing.models import BayesianNetwork
   from bn_testing.conditionals import LinearConditional


   model = BayesianNetwork(
      dag=RandomizedPathSubGraph(n_visible_nodes=20),
      conditionals=LinearConditional(),
   )

See also the documentation of
:py:class:`bn_testing.dags.RandomizedDAG` for how to set up a
randomized DAG.


Hidden nodes
------------

Nodes can be marked as hidden by setting their :py:mod:`networkx` node
attribute :code:`is_hidden` to :code:`True`:

.. code-block:: python

   class PathGraphWithHidden(DAG):

      def make_dag(self):
         dag = nx.path_graph(
            n=100,
            create_using=nx.DiGraph
         )

         dag.nodes[5]['is_hidden'] = True
         dag.nodes[10]['is_hidden'] = True
         return dag


This can also be done using the helper 
:py:func:`~bn_testing.dags.DAG.mark_as_hidden`:

.. code-block:: python

   class PathGraphWithHidden(DAG):

      def make_dag(self):
         dag = nx.path_graph(
            n=100,
            create_using=nx.DiGraph
         )

         dag = self.mark_as_hidden(dag, [5, 10])
         return dag


.. note::

   For DAG generators deriving from
   :py:class:`~bn_testing.dags.RandomizedDAG`, the hidden variables do
   not need to be set in
   :py:func:`~bn_testing.dags.RandomizedDAG.make_dag` as this is done
   by the class automatically.


Fixed distributions
-------------------

Sometimes, some parts of the graphical model need to be fixed, like
some terms, source distributions, or noise. This can be done while
constructing the graph by attaching attributes to the nodes to be
fixed.


Terms
"""""

Specific terms can be attached to nodes during the graph building (for
instance, to hidden nodes):

.. code-block:: python

   from bn_testing.dags import DAG
   from bn_testing.terms import (
      Term,
      Linear,
   )
   import pymc as pm

   class PathGraph(DAG):

      def make_dag(self):
         dag = nx.path_graph(
            n=self.n_nodes,
            create_using=nx.DiGraph
         )

         # Adding a visible node
         dag.add_node(
            'V',
            term=Linear(parents=[0, 1], coefs=[1, -1]),
            noise=pm.Normal.dist(mu=0, sigma=0.1)
         )
         dag.add_edges_from([(0, 'V'), (1, 'V')])

         # Add a hidden node 
         dag.add_node(
            'H',
            term=Term(
               parents=[0],
               term_fn=lambda v: 2*np.sqrt(v[0])
            ),
            no_noise=True,
            hidden=True

         )
         dag.add_edges_from([(0, 'H'), ('H', 1)])

         return dag



Source distributions
""""""""""""""""""""

Another scenario is to fix the distribution of some source nodes. This
can be done setting the `distribution` attribute of the nodes:

.. code-block:: python

   from bn_testing.dags import DAG
   import pymc as pm

   class PathGraph(DAG):

      def make_dag(self):
         dag = nx.path_graph(
            n=self.n_nodes,
            create_using=nx.DiGraph
         )

         mu = self.random.uniform(-1, 1)
         dag.nodes[0]['distribution'] = pm.Normal.dist(mu=mu, sigma=0.1)
         return dag

.. note::
   
   The attribiute `distribution` is ignored for any non-source node
   and vice-verse are the attributes `noise` and `term` for nodes with
   incoming edges. 
