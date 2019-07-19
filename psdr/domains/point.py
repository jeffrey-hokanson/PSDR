from __future__ import division

import numpy as np

from .domain import TOL
from .box import BoxDomain

class PointDomain(BoxDomain):
	r""" A domain consisting of a single point

	Given a point :math:`\mathbf{x} \in \mathbb{R}^m`, construct the domain consisting of that point

	.. math::
	
		\mathcal{D} = \lbrace \mathbf x \rbrace \subset \mathbb{R}^m.

	Parameters
	----------
	x: array-like (m,)
		Single point to be contained in the domain
	"""
	def __init__(self, x, names = None):
		self._point = np.array(x).flatten()
		self._init_names(names)	
		assert len(self._point.shape) == 1, "Must provide a one-dimensional point"

	def __len__(self):
		return self._point.shape[0]
		
	def closest_point(self, x0):
		return np.copy(self._point)

	def _corner(self, p):
		return np.copy(self._point)

	def _extent(self, x, p):
		return 0

	def _isinside(self, X, tol = TOL):
		Pcopy = np.tile(self._point.reshape(1,-1), (X.shape[0],1))
		return np.all(X == Pcopy, axis = 1)	

	def _sample(self, draw = 1):
		return np.tile(self._point.reshape(1,-1), (draw, 1))

	def is_point(self):
		return True

	@property
	def lb(self):
		return np.copy(self._point)

	@property
	def ub(self):
		return np.copy(self._point)
