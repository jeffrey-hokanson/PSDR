====================================
Coordinate-based Dimension Reduction
====================================

The goal of a coordinate based dimension reduction is to identify
subsets of variables that are sufficient to explain the behavor of the function;
this process is sometimes known as *variable screening*.

Although in many ways this is a simpler process than subspace based dimension reduction,
from the point of view of the code, a coordinate-based dimension reduction is a 
special case of a subspace-based dimension reduction where subspaces are built from
columns of the identity matrix.

.. autoclass:: psdr.CoordinateBasedDimensionReduction
	:members:

Diagonal Lipschitz Matrix
=========================

.. autoclass:: psdr.DiagonalLipschitzMatrix

