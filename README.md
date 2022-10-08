# BN testing

[![Documentation Status](https://readthedocs.org/projects/bn_testing/badge/?version=latest)](https://bn_testing.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/bn_testing)](https://pypi.org/project/bn_testing/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bn_testing)

A test framework to evaluate methods that learn Bayesian Networks from
high-dimensional observed data.


## Sampling

Set up the graphical model and sample  data
```python
from bn_testing.models import BayesianNetwork
from bn_testing.dags import ErdosReny
from bn_testing.conditionals import PolynomialConditional


model = BayesianNetwork(
   n_nodes=100,
   dag=ErdosReny(p=0.01),
   conditionals=PolynomialConditional(max_terms=5)
)

df = model.sample(10000)
```
The observations are stored in a `pandas.DataFrame` where the columns
are the nodes of the DAG and each row is an observation. The
underlying DAG of the graphical model can be accessed with `model.dag`
