""" Descriptions of various bases"""

__all__ = ['PolynomialTensorBasis', 
	'MonomialTensorBasis', 
	'LegendreTensorBasis',
	'ChebyshevTensorBasis',
	'LaguerreTensorBasis',
	'HermiteTensorBasis',
 ]

import numpy as np
from numpy.polynomial.polynomial import polyvander, polyder
from numpy.polynomial.legendre import legvander, legder, legroots 
from numpy.polynomial.chebyshev import chebvander, chebder
from numpy.polynomial.hermite import hermvander, hermder
from numpy.polynomial.laguerre import lagvander, lagder

class Basis(object):
	pass



################################################################################
# Indexing utility functions for total degree
################################################################################

#TODO: Should these be moved into PolynomialTensorBasis class? 

def _full_index_set(n, d):
	""" A helper function for index_set.
	"""
	if d == 1:
		I = np.array([[n]])
	else:
		II = _full_index_set(n, d-1)
		m = II.shape[0]
		I = np.hstack((np.zeros((m, 1)), II))
		for i in range(1, n+1):
			II = _full_index_set(n-i, d-1)
			m = II.shape[0]
			T = np.hstack((i*np.ones((m, 1)), II))
			I = np.vstack((I, T))
	return I

def index_set(n, d):
	"""Enumerate multi-indices for a total degree of order `n` in `d` variables.
	
	Parameters
	----------
	n : int
		degree of polynomial
	d : int
		number of variables, dimension
	Returns
	-------
	I : ndarray
		multi-indices ordered as columns
	"""
	I = np.zeros((1, d), dtype = np.int)
	for i in range(1, n+1):
		II = _full_index_set(i, d)
		I = np.vstack((I, II))
	return I[:,::-1].astype(int)

#class MultiIndex:
#	"""Specifies a multi-index for a polynomial in the monomial basis of fixed total degree 
#
#	"""
#	def __init__(self, dimension, degree):
#		self.dimension = dimension
#		self.degree = degree
#		#self.iterator = product(range(0, degree+1), repeat = dimension)	
#		self.idx = index_set(degree, dimension)
#		self.iterator = iter(self.idx)
#
#	def __iter__(self):
#		return self
#
#	def next(self):
#		return self.iterator.next()
#
#	def __len__(self):
#		return int(comb(self.degree + self.dimension, self.degree, exact = True))


class PolynomialTensorBasis(Basis):
	r""" Generic tensor product basis of fixed total degree

	This class constructs a tensor product basis of dimension :math:`n`
	of fixed given degree :math:`p` given a basis for polynomials
	in one variable. Namely, this basis is composed of elements:

	.. math::

		\psi_j(\mathbf x) := \prod_{i=1}^n \phi_{[\boldsymbol \alpha_j]_i}(x_i) 
			\quad \sum_{i=1}^n [\boldsymbol \alpha_j]_i \le p;
			\quad \phi_i \in \mathcal{P}_{i}(\mathbb{R})


	Parameters
	----------
	n: int
		The input dimension of the space
	p: int
		The total degree of polynomials
	polyvander: function
		Function providing the scalar Vandermonde matrix (i.e., numpy.polynomial.polynomial.polyvander)
	polyder: function
		Function providing the derivatives of scalar polynomials (i.e., numpy.polynomial.polynomial.polyder)	
 
	"""
	def __init__(self, n, p, polyvander, polyder):
		self.n = n
		self.p = p
		self.vander = polyvander
		self.der = polyder
		self.indices = index_set(p, n).astype(int)
		self._build_Dmat()

	def _build_Dmat(self):
		""" Constructs the (scalar) derivative matrix
		"""
		self.Dmat = np.zeros( (self.p+1, self.p))
		for j in range(self.p + 1):
			ej = np.zeros(self.p + 1)
			ej[j] = 1.
			self.Dmat[j,:] = self.der(ej)

	def V(self, X):
		r""" Builds the Vandermonde matrix associated with this basis

		Given points :math:`\mathbf x_i \in \mathbb{R}^n`, 
		this creates the Vandermonde matrix

		.. math::

			[\mathbf{V}]_{i,j} = \phi_j(\mathbf x_i)

		where :math:`\phi_j` is a multivariate polynomial as defined in the class definition.

		Parameters
		----------
		X: array-like (M, n)
			Points at which to evaluate the basis at where :code:`X[i]` is one such point in 
			:math:`\mathbf{R}^n`.
		
		Returns
		-------
		V: np.array
			Vandermonde matrix
		"""
		X = np.array(X)
		M = X.shape[0]
		V_coordinate = [self.vander(X[:,k], self.p) for k in range(self.n)]
		
		V = np.ones((M, len(self.indices)), dtype = X.dtype)
		
		for j, alpha in enumerate(self.indices):
			for k in range(self.n):
				V[:,j] *= V_coordinate[k][:,alpha[k]]
		return V

	def VC(self, X, c):
		r""" Evaluate the product of the Vandermonde matrix and a vector

		This evaluates the product :math:`\mathbf{V}\mathbf{c}`
		where :math:`\mathbf{V}` is the Vandermonde matrix defined in :code:`V`.
		This is done without explicitly constructing the Vandermonde matrix to save
		memory.	
		 
		Parameters
		----------
		X: array-like (M,n)
			Points at which to evaluate the basis at where :code:`X[i]` is one such point in 
			:math:`\mathbf{R}^n`.
		c: array-like 
			The vector to take the inner product with.
		
		Returns
		-------
		Vc: np.array (M,)
			Product of Vandermonde matrix and :math:`\mathbf c`
		"""
		X = np.array(X)
		M = X.shape[0]
		c = np.array(c)
		assert len(self.indices) == c.shape[0]

		if len(c.shape) == 2:
			oneD = False
		else:
			c = c.reshape(-1,1)
			oneD = True

		V_coordinate = [self.vander(X[:,k], self.p) for k in range(self.n)]
		out = np.zeros((M, c.shape[1]))	
		for j, alpha in enumerate(self.indices):

			# If we have a non-zero coefficient
			if np.max(np.abs(c[j,:])) > 0.:
				col = np.ones(M)
				for ell in range(self.n):
					col *= V_coordinate[ell][:,alpha[ell]]

				for k in range(c.shape[1]):
					out[:,k] += c[j,k]*col
		if oneD:
			out = out.flatten()
		return out

	def DV(self, X):
		r""" Column-wise derivative of the Vandermonde matrix

		Given points :math:`\mathbf x_i \in \mathbb{R}^n`, 
		this creates the Vandermonde-like matrix whose entries
		correspond to the derivatives of each of basis elements;
		i.e., 

		.. math::

			[\mathbf{V}]_{i,j} = \phi_j'(\mathbf x_i)

		Parameters
		----------
		X: array-like (M, n)
			Points at which to evaluate the basis at where :code:`X[i]` is one such point in 
			:math:`\mathbf{R}^n`.

		Returns
		-------
		Vp: np.array
			Column-wise derivative of Vandermonde matrix
		"""
		X = np.array(X)
		M = X.shape[0]
		V_coordinate = [self.vander(X[:,k], self.p) for k in range(self.n)]
		
		#mi = MultiIndex(self.n, self.p)
		#N = len(mi)
		N = len(self.indices)
		DV = np.ones((M, N, self.n), dtype = X.dtype)

		for k in range(self.n):
			for j, alpha in enumerate(self.indices):
				for q in range(self.n):
					if q == k:
						DV[:,j,k] *= np.dot(V_coordinate[q][:,0:-1], self.Dmat[alpha[q],:])
					else:
						DV[:,j,k] *= V_coordinate[q][:,alpha[q]]

		return DV


class MonomialTensorBasis(PolynomialTensorBasis):
	"""A tensor product basis of bounded total degree built from the monomials"""
	def __init__(self, n, p):
		PolynomialTensorBasis.__init__(self, n, p, polyvander, polyder)

class LegendreTensorBasis(PolynomialTensorBasis):
	"""A tensor product basis of bounded total degree built from the Legendre polynomials

	"""
	def __init__(self, n, p):
		PolynomialTensorBasis.__init__(self, n, p, legvander, legder)

class ChebyshevTensorBasis(PolynomialTensorBasis):
	"""A tensor product basis of bounded total degree built from the Chebyshev polynomials
	
	"""
	def __init__(self, n, p):
		PolynomialTensorBasis.__init__(self, n, p, chebvander, chebder)

class LaguerreTensorBasis(PolynomialTensorBasis):
	"""A tensor product basis of bounded total degree built from the Laguerre polynomials

	"""
	def __init__(self, n, p):
		PolynomialTensorBasis.__init__(self, n, p, lagvander, lagder)

class HermiteTensorBasis(PolynomialTensorBasis):
	"""A tensor product basis of bounded total degree built from the Hermite polynomials

	"""
	def __init__(self, n, p):
		PolynomialTensorBasis.__init__(self, n, p, hermvander, hermder)





