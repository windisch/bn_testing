DAG generators
==============



Available generators
--------------------


* :py:class:`~bn_testing.dags.ScaleFree`: Generates a scale-free dag
* :py:class:`~bn_testing.dags.ErdosReny`: Generates a DAG based on the
  Erdos-Reny model


Implementing own generators
---------------------------

.. code-block:: python

   from bn_testing.dags import DAG

   class PathGraph(DAG):

      def generate(self):
         # Generate a dag using  self.n_nodes
         dag = nx.path_graph(
            n=self.n_nodes,
            create_using=nx.DiGraph
         )

         # Optionally, attach some fixed terms
         dag.nodes[1]['term'] = Linear(parents=[0], coefs=[10])

         return dag
