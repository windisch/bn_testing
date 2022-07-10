# BN testing

A test framework to evaluate methods that learn Bayesian Networks from
high-dimensional observed data.


## Sampling

Set up the graphical model and sample  data
```python
from bn_testing.dags import GroupedGaussianBN

model = GroupedGaussianBN(n_nodes=200, n_groups=10)
df = model.sample(10000)
```
The observations are stored in a `pandas.DataFrame` where the columns
are the nodes of the DAG and each row is an observation. The
underlying DAG of the graphical model can be accessed with `model.dag`
