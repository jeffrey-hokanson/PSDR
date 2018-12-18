Domain
======


Design Considerations
----------------------
In order to use dimension reduction for practical algorithms,
we need be able to solve three problems:

=========================            ==============================================================================================
Problem                              Mathematical statement
=========================            ==============================================================================================
extent                               :math:`\displaystyle \max_{\alpha \ge 0} \alpha \text{ such that } \mathbf{x}_0+\alpha \mathbf{p}\in \mathcal{D}`
closest point                        :math:`\displaystyle \min_{\mathbf x \in \mathcal D} \| \mathbf L (\mathbf x - \mathbf y)\|_2`
furthest point (corner)              :math:`\displaystyle \max_{\mathbf x \in \mathcal D} \mathbf p^\top \mathbf x`
constrained least squares            :math:`\displaystyle \min_{\mathbf x \in \mathcal D} \| \mathbf A \mathbf x - \mathbf b \|_2`
=========================            ==============================================================================================

With these three operations, we can, for example,
implement hit-and-run sampling.



Abstract Base Class
-------------------

All domains implement a similar interface that provides these operations.

.. autoclass:: psdr.Domain
   :members:


Deterministic Domains
---------------------
Here we use deterministic domains to describe domains
whose main function is to specify a series of constraints.
These classes do have a :code:`sample` method,
but this sampling is simply random with uniform probability over the domain. 
The classes below are in the nesting order;
i.e., a :code:`BoxDomain` is a subset of a :code:`LinIneqDomain`.
These distinctions are important as we can often use less expensive algorithms 
for each of the subclasses.


.. autoclass:: psdr.LinQuadDomain

.. autoclass:: psdr.LinIneqDomain
   :members: chebyshev_center 

.. autoclass:: psdr.ConvexHullDomain

.. autoclass:: psdr.BoxDomain

.. autoclass:: psdr.PointDomain




Random Domains
--------------
An alternative function of domains is to provide
samples from an associate sampling measure on some domain :math:`\mathcal{D}`.
Domains with a stochastic interpretation are all subclasses of :code:`psdr.RandomDomain`
and implement several additional functions.

.. autoclass:: psdr.RandomDomain
   :members:

.. autoclass:: psdr.UniformDomain

.. autoclass:: psdr.NormalDomain

.. autoclass:: psdr.LogNormalDomain


Tensor Product Domain
---------------------
As part of the operations on domains, 
one goal is to compose a tensor product of sub-domains;
i.e., given :math:`\mathcal{D}_1` and :math:`\mathcal{D}_2`
to form:

.. math::
	
	\mathcal{D} := \mathcal{D}_1 \otimes \mathcal{D}_2.


.. autoclass:: psdr.TensorProductDomain


Domain Operations
-----------------



