from __future__ import division

from .random import RandomDomain
from .box import BoxDomain

class UniformDomain(BoxDomain, RandomDomain):
	r""" A randomized version of a BoxDomain with a uniform measure on the space.
	"""
	
	def _pdf(self, x):
		return np.one(x.shape[0])/np.prod([(ub_ - lb_) for lb_, ub_ in zip(self.lb, self.ub)])
