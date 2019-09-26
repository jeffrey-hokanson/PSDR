r""" Synthetic demo functions based on polynomials 

""" 

import numpy as np
from psdr import Function, BoxDomain


class AffineFunction(Function):
	r""" Constructs an affine function

	Parameters
	"""
	def __init__(self, linear = None, constant = None, domain = None):
		if linear is None:
			if domain is not None:
				self.linear = np.random.randn(len(domain))
			else:
				self.linear = np.random.randn(5)
		else:
			self.linear = np.array(linear).flatten()

		if constant is None:
			self.constant = 0
		else:
			self.constant = float(constant)


		if domain is None:
			domain = BoxDomain(-np.ones(len(self.linear)), np.ones(len(self.linear)))
		

		Function.__init__(self, self._func, domain, grads = self._func_grad, vectorized = True)

	def _func(self, X):
		return self.linear.dot(X.T) + self.constant 

	def _func_grad(self, X):
		return np.tile(self.linear, (len(X),1))



class QuadraticFunction(Function):
	def __init__(self, quad = None, linear = None, constant = None, domain = None):
		
		# Determine dimension of space
		if quad is not None:
			m = len(quad)
		elif linear is not None:
			m = len(linear)
		elif domain is not None:
			m = len(domain)
		else:
			m = 5
		
		if quad is not None:
			self.quad = np.array(quad).reshape(m,m)
		else:
			self.quad = np.eye(m)

		if linear is not None:
			self.linear = np.array(linear).reshape(m)
		else:
			self.linear = np.zeros(m)

		if constant is not None:
			self.constant = float(constant)
		else:
			self.constant = 0.

		if domain is None:
			domain = BoxDomain(-np.ones(len(self.linear)), np.ones(len(self.linear)))
	
		Function.__init__(self, self._func, domain, grads = self._func_grad, vectorized = True)

	def _func(self, X):
		# Fast quadratic form computation
		# https://stackoverflow.com/a/18542236
		X = np.atleast_2d(X)
		return 	(X.dot(self.quad)*X).sum(axis = 1) + self.linear.dot(X.T) + self.constant

	def _func_grad(self, X):
		 return 2*X.dot(self.quad) + np.tile(self.linear, (len(X),1))
