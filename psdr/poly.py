
""" Polynomial functions """
# (c) 2019 Jeffrey M. Hokanson (jeffrey@hokanson.us)

import numpy as np
import scipy.linalg
from copy import copy
from basis import *
from function import BaseFunction

__all__ = ['PolynomialFunction', 'PolynomialApproximation']


def linear_fit(A, b, norm = 2, bound = None):
	r""" solve the linear optimization problem subject to constraints
	"""
	assert norm in [1,2,np.inf], "Invalid norm specified"
	assert bound in [None, 'lower', 'upper'], "invalid bound specified"

	if norm == 2 and bound == None:
		return scipy.linalg.lstsq(A, b)[0]
	else:
		x = cp.Variable(A.shape[1])
		residual = x.__rmatmul__(A) - b
		if norm == 1:   	 obj = cp.norm1(residual)
		elif norm == 2: 	 obj = cp.norm(residual)
		elif norm == np.inf: obj = cp.norm_inf(residual)
		
		if bound == 'lower':
			constraint = [residual <= 0]
		elif bound == 'upper':
			constraint = [residual >= 0]
		else:
			constraint = []
		
		# Now actually solve the problem
		problem = cp.Problem(cp.Minimize(obj), constraint)
		problem.solve(feastol = 1e-10, solver = cp.ECOS)
		return x.value


class PolynomialFunction(BaseFunction):
	def __init__(self, dimension, degree, coef):
		self.basis = LegendreTensorBasis(dimension, degree) 
		self.coef = coef

	def eval(self, X):
		V = self.basis.V(X)
		return V.dot(self.coef) 

	def grad(self, X):
		return np.tensordot(self.basis.DV(X), self.coef, axes = (1,0))

	def hessian(self, X):
		return np.tensordot(self.basis.DDV(X), self.coef, axes = (1,0))
	

class PolynomialApproximation(PolynomialFunction):
	def __init__(self, degree, basis = 'legendre', norm = 2, bound = None):

		degree = int(degree)
		assert degree >= 0, "Degree must be positive"
		self.degree = degree

		assert basis in ['legendre', 'monomial', 'chebyshev', 'laguerre', 'hermite']
		self.basis_name = copy(basis)

		self.basis = None
		
		assert bound in [None, 'lower', 'upper']
		self.bound = bound

		assert norm in [1,2, np.inf]
		self.norm = norm
		

	def fit(self, X, fX):
		M, m = X.shape

		# Since we don't know the input dimension until we get the data, we initialize the basis here
		if self.basis_name == 'legendre':
			self.basis = LegendreTensorBasis(m, self.degree) 
		elif self.basis_name == 'monomial':
			self.basis = MonomialTensorBasis(m, self.degree) 
		elif self.basis_name == 'chebyshev':
			self.basis = ChebyshevTensorBasis(m, self.degree) 
		elif self.basis_name == 'laguerre':
			self.basis = LaguerreTensorBasis(m, self.degree) 
		elif self.basis_name == 'hermite':
			self.basis = HermiteTensorBasis(m, self.degree) 
		else:
			raise NotImplementedError

		# Scale the basis to the problem
		self.basis.set_scale(X)
		
		V = self.basis.V(X)

		self.coef = linear_fit(V, fX, norm = self.norm, bound = self.bound)

if __name__ == '__main__':
	X = np.random.randn(100, 5)
	fX = np.random.randn(100,)

	poly = PolynomialApproximation(degree = 2)
	poly.fit(X, fX)
	print poly.eval(X).shape
	print poly.grad(X).shape
