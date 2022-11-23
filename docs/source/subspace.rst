==================================
Subspace-based Dimension Reduction
==================================

Here our goal is to find a subspace along which our quantity of interest varies the most,
for various definitions of *most*.

.. autoclass:: psdr.SubspaceBasedDimensionReduction
   :members:


Active Subspace
===============

.. autoclass:: psdr.ActiveSubspace
	:members: fit


Outer Product Gradient
======================

.. autoclass:: psdr.OuterProductGradient
	:members: fit


Lipschitz Matrix
================

.. autoclass:: psdr.LipschitzMatrix
   :members: fit, H, L, uncertainty, uncertainty_domain, shadow_uncertainty 

.. autoclass:: psdr.LowRankLipschitzMatrix
   :members: alpha, J, U
