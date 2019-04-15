# PSDR: Parameter Space Dimension Reduction Toolbox
[![Documentation Status](https://readthedocs.org/projects/psdr/badge/?version=latest)](https://psdr.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/jeffrey-hokanson/PSDR.svg?branch=master)](https://travis-ci.org/jeffrey-hokanson/PSDR)
[![Coverage Status](https://coveralls.io/repos/github/jeffrey-hokanson/PSDR/badge.svg?branch=master)](https://coveralls.io/github/jeffrey-hokanson/PSDR?branch=master)
Author: Jeffrey M. Hokanson (jeffrey@hokanson.us)


## Introduction
Given a function mapping some subset of an m-dimensional space to a scalar value

![f: D subset R^m to R](eqn1.png) 

*parameter space dimension reduction* seeks to identify a low-dimensional manifold
of the input along which this function varies the most.
Frequently we will choose to use a linear manifold
and consequently identify linear combinations of input variables along 
which the function varies the most.

We emphasize that this library is for parameter space dimension reduction
as the term 'dimension reduction' often appears in other contexts.
For example, model reduction is often referred to as dimension reduction
because it reduces the state-space dimension of a set of differential equations,
yielding a smaller set of differential equations.

## Simple example

```python
import psdr, psdr.demos
fun = psdr.demos.Borehole()    # load a test problem
X = fun.domain.sample(1000)    # sample points from the domain with uniform probabilty
grads = fun.grad(X)            # evaluate the gradient at the points in X
act = psdr.ActiveSubspace()    # initialize a class to find the Active Subspace
act.fit(grads)                 # estimate the active subspace using these Monte-Carlo samples
print(act.U[:,0])              # print the most important linear combination of variables

>>> array([ 9.19118904e-01, -2.26566967e-03,  2.90116247e-06,  2.17665629e-01,
        2.78485430e-03, -2.17665629e-01, -2.21695479e-01,  1.06310937e-01])
```


## Documentation
For further documentation, please see our page on Read the Docs:
[Documentation](https://psdr.readthedocs.io/en/latest/).


## Contributing




