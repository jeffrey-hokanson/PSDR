""" Ridge-approximation based optimization"""


import numpy as np
from function import Function
from polyridge import PolynomialRidgeApproximation 


class RidgeOptimization:
	r""" Ridge-based nonlinear optimization

	Given a vector valued function :math:`\mathbf{f}: \mathcal{D}\subset \mathbb{R}^m\to \mathbb{R}^n`,
	this class solves the optimization problem

	.. math::

		\min_{\mathbf{x}\in \mathcal{D}} &\ f_0(\mathbf{x}) \\
		\text{such that} & \ f_i(\mathbf{x}) \le 0

	by constructing a sequence of bounding surrogates.
	[This sign convention follows Ch. 5, [BV04]_ ]


	Parameters
	----------
	func : Function
		Function to optimize	

	References
	----------
	.. [BV04] Convex Optimization. Stephen Boyd and Lieven Vandenberghe. Cambridge University Press. 2004. 

	"""
	def __init__(self, func, X = None, fX = None, pool = None):
		self.func = func
		self.domain = func.domain_norm 
		self.X = []
		self.fX = []

	def step(self):

		pass

if __name__ == '__main__':
	from demos import golinksi_volume, golinksi_design_domain

	domain = golinksi_design_domain()
	func = Function(golinksi_volume, domain)

	x = func.sample(1)
	print func(x)
	pass
