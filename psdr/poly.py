
""" Polynomial functions """
# (c) 2019 Jeffrey M. Hokanson (jeffrey@hokanson.us)

from __future__ import print_function
import numpy as np
import cvxpy as cp
import scipy.linalg
import cvxpy as cp
from copy import copy
from .basis import *
from .function import BaseFunction


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
		problem.solve(feastol = 1e-10, reltol = 1e-8, abstol = 1e-8, solver = cp.ECOS)
		return x.value


class PolynomialFunction(BaseFunction):
	r""" A polynomial function in a Legendre basis 	


	Parameters
	----------
	dimension: int
		Input dimension
	degree: int
		Degree of polynomial
	coef: array-like
		Coefficients of polynomial	
	"""
	def __init__(self, dimension, degree, coef):
		self.basis = LegendreTensorBasis(dimension, degree) 
		self.coef = coef

	def V(self, X):	
		return self.basis.V(X)

	def DV(self, X):
		return self.basis.DV(X)

	def DDV(self, X):
		return self.basis.DDV(X)

	def eval(self, X):
		if len(X.shape) == 1:
			return self.V(X.reshape(1,-1)).dot(self.coef).reshape(1)
		else:
			return self.V(X).dot(self.coef)

	def grad(self, X):
		if len(X.shape) == 1:
			one_d = True
			X = X.reshape(1,-1)	
		else:
			one_d = False	
		
		DV = self.DV(X)
		# Compute gradient on projected space
		Df = np.tensordot(DV, self.coef, axes = (1,0))
		# Inflate back to whole space
		if one_d:
			return Df.reshape(X.shape[1])
		else:
			return Df

	def hessian(self, X):
		if len(X.shape) == 1:
			one_d = True
			X = X.reshape(1,-1)	
		else:
			one_d = False
	
		DDV = self.DDV(X)
		DDf = np.tensordot(DDV, self.coef, axes = (1,0))
		if one_d:
			return DDf.reshape(X.shape[1], X.shape[1])
		else:
			return DDf
	

class PolynomialApproximation(PolynomialFunction):
	r""" Construct a polynomial approximation

	Parameters
	----------
	degree: int
		Degree of polynomial
	basis: ['legendre', 'monomial', 'chebyshev', 'laguerre', 'hermite']
		Basis in which to express the polynomial
	norm: [1, 2, np.inf]
		Norm in which to find the approximation
	bound: [None, 'lower', 'upper']
		If None, construct approximation in the specified norm;
		if 'lower' or 'upper', additionally enforce the constraint that
		the approximation is below or above the measured samples	
	"""
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

