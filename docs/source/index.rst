bn_testing
==========

A test framework to evaluate methods that learn Bayesian Networks from
high-dimensional observed data. It provides helpers to construct
Bayesian networks in a randomized fashion and helps sampling
observational data from it. Its **not** a framework to fit Bayesian
networks on data!


.. note::

   Currently, only additive models are supported.

Quick start
-----------

Set up the graphical model and sample  data

.. code-block:: python

   from bn_testing.models import BayesianNetwork
   from bn_testing.dags import ErdosReny
   from bn_testing.conditionals import PolynomialConditional

   model = BayesianNetwork(
      n_nodes=100,
      dag=ErdosReny(p=0.01),
      conditionals=PolynomialConditional(max_terms=5)
   )
   
   df = model.sample(10000)

The observations are stored in a :py:class:`pandas.DataFrame` where the columns
are the nodes of the DAG and each row is an observation. The
underlying DAG of the graphical model can be accessed with `model.dag`

.. note::

   This project is under active development.



Contents
--------

.. toctree::

   install
   dags
   conditionals
   terms
   interventions
   examples
   api
