Interventions
=============

.. _interventions:

.. role:: python(code)
   :language: python


Given a :py:class:`bn_testing.models.Model` object, modifications,
like interventions, of the terms and source distributions can be
applied and data from the modified model can be sampled.

The goal of these modificiations may be:

* Generating interventional data.
* Simulation of changed behavior in the system under study.


Apply Modifications
-------------------

Assume that :code:`model` represents a additive structural equation
model with DAG :math:`G` and equations

.. math::
   X_i = f_i(X_j, j\in\text{pa}_G(i)) + \epsilon_i


Generally, modifications are changes of the transformations
:math:`f_i` and the additive noise :math:`\epsilon_i`.

Assume we have given a :code:`model` having a node Assume a node
:math:`X_3` with parents :math:`X_1` and :math:`X_2` and :math:`f_3(x_1,
x_2)=x_1^2\cdot x_2+x_1` and :math:`\epsilon_3\sim\mathcal{N}(0, 1)`.
That means, 

.. math::

   X_3 = X_1^2\cdot X_2+X_1+\epsilon_3

Randomized modification
"""""""""""""""""""""""

One way to change the construction of :math:`X_3` from its parents is
by randomly generating a new  :py:class:`~bn_testing.terms.Term`
and additive noise object using the :py:class:`~bn_testing.conditionals.Conditional` used
when instantiating the model. This can be done as folllows:

.. code-block:: python

   model.modify_node(node='X3')


The same works for source nodes, where a new source distribution is
generated using the
:py:func:`~bn_testing.conditionals.Conditional.make_source` method of
the condidtional object.

.. note::

   The model modification is done inplace, that means, an applied
   modification cannot be undone. 


Alternatively, the new term can also be generated using another
conditional: 

.. code-block:: python

   from bn_testing.conditinoals import LinearConditional

   model.modify_node(node='X3', conditionals=LinearConditional())


.. note::

   A modification not only changes the term, but also the additive
   noise!


Controlled modification
"""""""""""""""""""""""

Inner nodes
^^^^^^^^^^^

Instead of randomly generating a new term for :math:`X_3` is constructed,
we can also directly set the new term:

.. code-block:: python

   from bn_testing.terms import Linear
   import pymc as pm

   model.modify_inner_node(
      node='X3', 
      term=Linear(['X1', 'X2'], [0.5, -0.1],
      noise=pm.Beta.normal(alpha=0.2, beta=0.1),
    )


Note that if either the :code:`noise` or the :code:`term` are omitted,
a respective object is generated using the
:py:class:`~bn_testing.conditionals.Conditional` given when
instantiating the model.


Source nodes
^^^^^^^^^^^^

If :math:`X1` is a source node, its distribution can be replaced with
another distribution as follows:


.. code-block:: python

   import pymc as pm

   model.modify_source_node(
      node='X3',
      distribution=pm.Beta.dist(alpha=0.9, beta=0.1),
   )


Computing causal effects
------------------------

A special case of a modification is when a random variable is set to a
constant value. Typically, this is done because the average causal effect of
:math:`X_i` having value :math:`k` onto another variable, like
:math:`X_j` should be determined. This is

.. math::

   \mathbb{E}[X_j|X_i=k] - \mathbb{E}[X_j]

In our example above, setting :math:`X_1=2` permantently can
be archived with:

.. code-block:: python

   from bn_testing.conditionals import ConstantConditional
   import pymc as pm

   model.modify_node(
      node='X1', 
      conditionals=ConstantConditional(2)
    )


The average causal effect for this setting can be computed using the
shortcut :py:func:`~bn_testing.models.Model.compute_average_causal_effect`:


.. code-block:: python

   model.compute_average_causal_effect(
      node_from='X1',
      node_onto='X3',
      value=2,
   )


.. note::

   The calculation of the expected values used in the formula of the
   average causal effect is done empirically by sampling from the
   model with and without modification. The number of samples used can
   be changed by setting the paramater :code:`n`.
