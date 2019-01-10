"""Ridge function approximation from function values"""
# (c) 2017 Jeffrey M. Hokanson (jeffrey@hokanson.us)

import numpy as np
import scipy.linalg
from copy import deepcopy, copy

from domains import Domain, BoxDomain
from function import Function
from subspace import SubspaceBasedDimensionReduction
from basis import *
from opt import gauss_newton 


class PolynomialRidgeFunction(Function, SubspaceBasedDimensionReduction):
	r""" A polynomial ridge function
	"""
	def __init__(self, basis, coef, U):
		self.basis = basis
		self.coef = np.copy(coef)
		self._U = np.array(U)
		self.domain = None
		self.scale = False

	@property
	def U(self):
		return self._U

	# These functions allow scaling on the interior to remove conditioning issues
	def _UX(self, X, U = None):
		""" Evaluate the product Y = (U.T@X.T).T and scale appropriately 
		"""
		if U is None: U = self.U
		Y = np.dot(U.T, X.T).T
		if self.scale:
			if isinstance(self.basis, HermiteTensorBasis):
				Y = (Y - self._mean[None,:])/self._std[None,:]/np.sqrt(2)
			else:
				Y = 2*(Y-self._lb[None,:])/(self._ub[None,:] - self._lb[None,:]) - 1
		return Y
	
	def _set_scale(self, X, U = None):
		""" Set the normalization map
		"""
		if U is None: U = self.U

		if self.scale:
			Y = np.dot(U.T, X.T).T
			if isinstance(self.basis, HermiteTensorBasis):
				self._mean = np.mean(Y, axis = 0)
				self._std = np.std(Y, axis = 0)
				# In numpy, 'hermite' is physicist Hermite polynomials
				# so we scale by 1/sqrt(2) to convert to the 'statisticians' Hermite 
				# polynomials which are orthogonal with respect to the standard normal
				#Y = (Y - mean[None,:])/std[None,:]/np.sqrt(2)
			else:	
				self._lb = np.min(Y, axis = 0)
				self._ub = np.max(Y, axis = 0)

	def eval(self, X):
		#Y = np.dot(U.T, X.T).T
		Y = self._UX(X)
		V = self.basis.V(Y)
		return V.dot(self.coef)
		#return self.basis.VC(Y, self.coef)
	
	def grad(self, X):
		#Y = np.dot(U.T, X.T).T
		Y = self._UX(X)
		
		Vp = self.basis.DV(Y)
		# Compute gradient on projected space
		Df = np.tensordot(Vp, self.coef, axes = (1,0))
		# Inflate back to whole space
		Df = Df.dot(U.T)
		return Df

	def roots(self):
		r""" Compute the roots of the polynomial
		"""
		raise NotImplementedError

	def derivative_roots(self):
		r""" Compute the roots of the derivative of the polynomial
		"""
		raise NotImplementedError

	# TODO: add plotting of the response surface
	#def shadow_plot(self, X, fX, dim = 1, ax = None):


################################################################################
# Two types of custom errors raised by PolynomialRidgeApproximation
################################################################################
class UnderdeterminedException(Exception):
	pass

class IllposedException(Exception):
	pass


def orth(U):
	""" Orthgonalize, but keep directions"""
	U, R = np.linalg.qr(U, mode = 'reduced')
	U = np.dot(U, np.diag(np.sign(np.diag(R)))) 
	return U

		



class PolynomialRidgeApproximation(PolynomialRidgeFunction):
	r""" Constructs a ridge approximation using a total degree approximation

	Given a basis of total degree polynomials :math:`\lbrace \psi_j \rbrace_{j=1}^N`
	on :math:`\mathbb{R}^n`, this class constructs a polynomial ridge function 
	that minimizes the mismatch on a set of points :math:`\lbrace \mathbf{x}_i\rbrace_{i=1}^M \subset \mathbb{R}^m`
	in a :math:`p`-norm:

	.. math::

		\min_{\mathbf{U} \in \mathbb{R}^{m\times n}, \  \mathbf{U}^\top \mathbf{U} = \mathbf I, \
			\mathbf{c}\in \mathbb{R}^N }
			\sqrt[p]{ \sum_{i=1}^M  \left|f(\mathbf{x}_i) - 
				\sum_{j=1}^N c_j \psi_j(\mathbf{U}^\top \mathbf{x}_i) \right|^p}

	This approach assumes :math:`\mathbf{U}` is an element of the Grassmann manifold
	obeying the orthogonality constraint.  

	For the 2-norm (:math:`p=2`) this implementation uses Variable Projection following [HC18]_ 
	to remove the solution of the linear coefficients :math:`\mathbf{c}`,
	leaving an optimization problem posed over the Grassmann manifold alone.

	For both the 1-norm and the :math:`\infty`-norm,
	this implementation uses ........

	Parameters
	----------
	degree: int
		Degree of polynomial

	subspace_dimension: int
		Dimension of the low-dimensional subspace associated with the ridge approximation.
	
	basis: ['legendre', 'monomial', 'chebyshev', 'laguerre', 'hermite']
		Basis for polynomial representation
	
	norm: [1, 2, np.inf, 'inf']
		Norm in which to evaluate the mismatch between the ridge approximation and the data	
	
	scale: bool (default:True)
		Scale the coordinates along the ridge to ameliorate ill-conditioning		
	

	References
	----------
	.. [HC18] J. M. Hokanson and Paul G. Constantine. Data-driven Polynomial Ridge Approximation Using Variable Projection. SIAM J. Sci. Comput. Vol 40, No 3, pp A1566--A1589, DOI:10.1137/17M1117690.
	"""

	def __init__(self, degree, subspace_dimension, basis = 'legendre', 
		norm = 2, n_init = 1, scale = True, keep_data = True, domain = None):

		assert isinstance(degree, int)
		assert degree >= 0
		self.degree = degree
			
		assert isinstance(subspace_dimension, int)
		assert subspace_dimension >= 1
		self.subspace_dimension = subspace_dimension

		if self.degree == 1 and subspace_dimension > 1:
			self.subspace_dimension = 1
		
		if self.degree == 0:
			self.subspace_dimension = 0

		basis = basis.lower()
		assert basis in ['legendre', 'monomial', 'chebyshev', 'laguerre', 'hermite']
		self.basis_name = copy(basis)

		if basis == 'legendre':
			self.basis = LegendreTensorBasis(self.subspace_dimension, self.degree) 
		elif basis == 'monomial':
			self.basis = MonomialTensorBasis(self.subspace_dimension, self.degree) 
		elif basis == 'chebyshev':
			self.basis = ChebyshevTensorBasis(self.subspace_dimension, self.degree) 
		elif basis == 'laguerre':
			self.basis = LaguerreTensorBasis(self.subspace_dimension, self.degree) 
		elif basis == 'hermite':
			self.basis = HermiteTensorBasis(self.subspace_dimension, self.degree) 
		else:
			raise NotImplementedError

		assert isinstance(keep_data, bool)
		self.keep_data = keep_data

		assert isinstance(scale, bool)
		self.scale = scale

		assert norm in [1,2,'inf', np.inf], "Invalid norm specified"
		if norm == 'inf': norm = np.inf
		self.norm = norm

		if domain is None:
			self.domain = None
		else:
			assert isinstance(domain, Domain)
			self.domain = deepcopy(domain)


	def fit(self, X, fX, U0 = None, **kwargs):
		r""" Given samples, fit the polynomial ridge approximation.

		Parameters
		----------
		X : array-like (M, m)
			Input coordinates
		fX : array-like (M,)
			Evaluations of the function at the samples
		
		"""
		
		X = np.array(X)
		fX = np.array(fX).flatten()	

		assert X.shape[0] == fX.shape[0], "Dimensions of input do not match"

		if U0 is not None:
			U0 = np.array(U0)
			assert U0.shape[0] == X.shape[1], "U0 has %d rows, expected %d based on X" % (U0.shape[0], X.shape[1])
			assert U0.shape[1] == self.subspace_dimension, "U0 has %d columns; expected %d" % (U0.shape[1], self.subspace_dimension)
			U0 = orth(U0)


		# TODO Implement multiple initializations
		if self.norm == 2:
			return self._fit_2_norm(X, fX, U0, **kwargs)
		else:
			raise NotImplementedError



	def _build_V(self, X, U):
		""" Build the Vandermonde matrix
		"""

		Y = self._UX(X, U)
		V = self.basis.V(Y)
		return V

	def _build_DV(self, X, U):
		if self.scale:
			if isinstance(self.basis, HermiteTensorBasis):
				d_scale = 1./self._std/np.sqrt(2)
			else:
				d_scale = 2./(self._ub - self._lb)
		else:
			d_scale = np.ones(U.shape[1])

		# normalized projected coordinates
		Y = self._UX(X, U)
		# Construct derivative matrix
		DV = self.basis.DV(Y)
		# Scale derivative matrix appropreatly
		for ell in range(len(d_scale)):
			DV[:,:,ell] *= d_scale[ell]

		return DV
		

	def _fit_fixed_U_2_norm(self, X, fX, U):
		self.U = orth(U)
		self._set_scale(X, U)
		V = self._build_V(X, U)
		self.coef = scipy.linalg.lstsq(V, fX)[0].flatten()



	def _fit_affine_2_norm(self, X, fX):
		XX = np.hstack([X, np.ones((X.shape[0],1))])
		b = scipy.linalg.lstsq(XX, fX)[0]
		U = b[0:-1].reshape(-1,1)
		return U	


	def _varpro_residual(self, X, fX, U_flat):
		U = U_flat.reshape(X.shape[1],-1)

		V = self._build_V(X, U)
		c = scipy.linalg.lstsq(V, fX)[0]
		r = fX - V.dot(c)
		return r

	def _varpro_jacobian(self, X, fX, U_flat):
		# Get dimensions
		M, m = X.shape
		U = U_flat.reshape(X.shape[1],-1)
		m, n = U.shape
		
		V = self._build_V(X, U)
		c = scipy.linalg.lstsq(V, fX)[0].flatten()
		r = fX - V.dot(c)
		DV = self._build_DV(X, U)
	
		Y, s, ZT = scipy.linalg.svd(V, full_matrices = False) 
	
		N = V.shape[1]
		J1 = np.zeros((M,m,n))
		J2 = np.zeros((N,m,n))

		for ell in range(n):
			for k in range(m):
				DVDU_k = X[:,k,None]*DV[:,:,ell]
				
				# This is the first term in the VARPRO Jacobian minus the projector out fron
				J1[:, k, ell] = DVDU_k.dot(c)
				# This is the second term in the VARPRO Jacobian before applying V^-
				J2[:, k, ell] = DVDU_k.T.dot(r) 

		# Project against the range of V
		J1 -= np.tensordot(Y, np.tensordot(Y.T, J1, (1,0)), (1,0))
		# Apply V^- by the pseudo inverse
		J2 = np.tensordot(np.diag(1./s),np.tensordot(ZT, J2, (1,0)), (1,0))
		J = -( J1 + np.tensordot(Y, J2, (1,0)))
		return J.reshape(J.shape[0], -1)

	def _trajectory(self, U_flat, Delta_flat, t):
		Delta = Delta_flat.reshape(-1, self.subspace_dimension)
		U = U_flat.reshape(-1, self.subspace_dimension)
		Y, s, ZT = scipy.linalg.svd(Delta, full_matrices = False, lapack_driver = 'gesvd')
		UZ = np.dot(U, ZT.T)
		U_new = np.dot(UZ, np.diag(np.cos(s*t))) + np.dot(Y, np.diag(np.sin(s*t)))
		U_new = orth(U_new).flatten()
		return U_new

	def _fit_2_norm(self, X, fX, U0, **kwargs):
		if U0 is None:
			U0 = self._fit_affine_2_norm(X, fX)
			if self.subspace_dimension > 1:
				U0 = np.hstack([U0, np.random.randn(X.shape[1], self.subspace_dimension-1)])
		
		# Setup scaling	
		self._set_scale(X, U = U0)
		
		# Define trajectory

		def gn_solver(J_flat, r):
			Y, s, ZT = scipy.linalg.svd(J_flat, full_matrices = False, lapack_driver = 'gesvd')
			# Apply the pseudoinverse
			n = self.subspace_dimension
			Delta_flat = -ZT[:-n**2,:].T.dot(np.diag(1/s[:-n**2]).dot(Y[:,:-n**2].T.dot(r)))
			return Delta_flat, s[:-n**2]

		def jacobian(U_flat):
			# set the scaling
			U = U_flat.reshape(X.shape[1],-1)
			self._set_scale(X, U = U0)
			return self._varpro_jacobian(X, fX, U_flat)

		def residual(U_flat):
			return self._varpro_residual(X, fX, U_flat)	

		U0_flat = U0.flatten() 
		U_flat, info = gauss_newton(residual, jacobian, U0_flat,
			trajectory = self._trajectory, gnsolver = gn_solver, **kwargs) 
		
		U = U_flat.reshape(-1, self.subspace_dimension)
		self._U = U
		# Find coefficients
		V = self._build_V(X, U)
		self.coef = scipy.linalg.lstsq(V, fX)[0].flatten()
	

class PolynomialRidgeBound(PolynomialRidgeFunction):
	pass



if __name__ == '__main__':
	p = 5
	m = 4
	n = 1
	M = 100

	U = orth(np.random.randn(m,n))
	coef = np.random.randn(p+1)
	prf = PolynomialRidgeFunction(LegendreTensorBasis(n,p), coef, U)

	X = np.random.randn(M,m)
	fX = prf.eval(X)

	U0 = orth(np.random.randn(m,n))
	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension  =n)
	pra.fit(X, fX)

	# TODO: Fix bug in scaling

#	print pra.U
#	print U
#
#	print pra.coef
#	print prf.coef
#	print pra(X) - prf(X)
	#V =	pra._build_V(X, U)
