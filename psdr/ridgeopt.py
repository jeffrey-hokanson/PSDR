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

	Parameters
	----------
	func : Function
		Function to optimize	

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
