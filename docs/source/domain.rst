Domain
======




Design Considerations
----------------------
The subclasses of :code:`Domain` all specify *closed convex* domains.
We restrict our attention to this class of domains
as we want all domains to have certain primative operations to be convex programs:

- Closest point :math:`\min_{\mathbf x \in \mathcal D} \| \mathbf L (\mathbf x - \mathbf y)\|_2`
- Projected closest point :math:`\min_{\mathbf x \in \mathcal D} \| \mathbf A \mathbf x - \mathbf b \|_2`
- The furthest point in a direction, a *corner* :math:`\max_{\mathbf x \in \mathcal D} \mathbf p^\top \mathbf x`

With these three operations we can then perform many operations

.. autoclass:: psdr.Domain
   :members:
