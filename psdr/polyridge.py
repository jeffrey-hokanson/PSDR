"""Ridge function approximation from function values"""
# (c) 2017 Jeffrey M. Hokanson (jeffrey@hokanson.us)

import numpy as np
import scipy.linalg
import cvxpy as cp
import warnings
from copy import deepcopy, copy

from domains import Domain, BoxDomain
from function import BaseFunction
from subspace import SubspaceBasedDimensionReduction
from ridge import RidgeFunction
from basis import *
from gn import gauss_newton 
from minimax import minimax

class PolynomialRidgeFunction(RidgeFunction):
	r""" A polynomial ridge function
	"""
	def __init__(self, basis, coef, U):
		self.basis = basis
		self.coef = np.copy(coef)
		self._U = np.array(U)
		self.domain = None
		self.scale = False

	
	def set_scale(self, X, U = None):
		""" Set the normalization map
		"""
		if U is None: U = self.U

		if self.scale:
			Y = np.dot(U.T, X.T).T
			self.basis.set_scale(Y)

	def V(self, X, U = None):
		if U is None: U = self.U
		X = np.array(X)	
		Y = U.T.dot(X.T).T
		return self.basis.V(Y)

	def DV(self, X, U = None):
		if U is None: U = self.U
		
		Y = U.T.dot(X.T).T
		return self.basis.DV(Y)

	def eval(self, X):
		V = self.V(X)
		return V.dot(self.coef)
	
	def grad(self, X):
		#Y = np.dot(U.T, X.T).T
		
		DV = self.DV(X)
		# Compute gradient on projected space
		Df = np.tensordot(DV, self.coef, axes = (1,0))
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

def inf_norm_fit(A, b):
	r""" Solve inf-norm linear optimization problem

	.. math::

		\min_{x} \| \mathbf{A} \mathbf{x} - \mathbf{b}\|_\infty

	"""
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', PendingDeprecationWarning)
		x = cp.Variable(A.shape[1])
		obj = cp.norm_inf(x.__rmatmul__(A) - b)
		problem = cp.Problem(cp.Minimize(obj))
		problem.solve()
		return x.value


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
	.. [HC18] J. M. Hokanson and Paul G. Constantine. 
		Data-driven Polynomial Ridge Approximation Using Variable Projection. 
		SIAM J. Sci. Comput. Vol 40, No 3, pp A1566--A1589, DOI:10.1137/17M1117690.
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
		elif self.norm == np.inf:
			return self._fit_inf_norm(X, fX, U0, **kwargs)
		else:
			raise NotImplementedError


	################################################################################	
	# Specialized Affine fits
	################################################################################	
	
	def _fit_affine_2_norm(self, X, fX):
		XX = np.hstack([X, np.ones((X.shape[0],1))])
		b = scipy.linalg.lstsq(XX, fX)[0]
		U = b[0:-1].reshape(-1,1)
		return U	


	def _fit_fixed_U_2_norm(self, X, fX, U):
		self.U = orth(U)
		self.set_scale(X)
		V = self.V(X)
		self.coef = scipy.linalg.lstsq(V, fX)[0].flatten()




	def _varpro_residual(self, X, fX, U_flat):
		U = U_flat.reshape(X.shape[1],-1)

		V = self.V(X, U)
		c = scipy.linalg.lstsq(V, fX)[0]
		r = fX - V.dot(c)
		return r

	def _varpro_jacobian(self, X, fX, U_flat):
		# Get dimensions
		M, m = X.shape
		U = U_flat.reshape(X.shape[1],-1)
		m, n = U.shape
		
		V = self.V(X, U)
		c = scipy.linalg.lstsq(V, fX)[0].flatten()
		r = fX - V.dot(c)
		DV = self.DV(X, U)
	
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
		self.set_scale(X, U = U0)
		
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
			self.set_scale(X, U = U)
			return self._varpro_jacobian(X, fX, U_flat)

		def residual(U_flat):
			return self._varpro_residual(X, fX, U_flat)	

		U0_flat = U0.flatten() 
		U_flat, info = gauss_newton(residual, jacobian, U0_flat,
			trajectory = self._trajectory, gnsolver = gn_solver, **kwargs) 
		
		U = U_flat.reshape(-1, self.subspace_dimension)
		self._U = U
		# Find coefficients
		V = self.V(X, U)
		self.coef = scipy.linalg.lstsq(V, fX)[0].flatten()

	def _fit_fixed_U_inf_norm(self, X, fX, U):
		self._U = U
		V = self.V(X, U)
		# Setup minimax problem for coeffiecients
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', PendingDeprecationWarning)
			c = cp.Variable(V.shape[1])
			obj = cp.norm_inf( fX - c.__rmatmul__(V))
			problem = cp.Problem(cp.Minimize(obj))
			problem.solve()
			self.coef = c.value

	def _inf_residual_grad(self, X, fX, U_c, return_grad = False):
		M, m = X.shape
		n = self.subspace_dimension
		N = len(self.basis.indices)
		U = U_c[:m*n].reshape(m,n)
		c = U_c[m*n:].reshape(N)

		V = self.V(X, U)
		
		res = V.dot(c) - fX

		if not return_grad:
			return res

		# Derivative of V with respect to U with c fixed	
		DVDUc = np.zeros((M,m,n))
		DV = self.DV(X, U) 	# Size (M, N, n)
		for k in range(m):
			for ell in range(n):
				DVDUc[:,k,ell] = X[:,k]*np.dot(DV[:,:,ell], c)
		grad = np.hstack([DVDUc.reshape(M,-1), V])
		return res, grad
		
	def _inf_trajectory(self, X, fX, U_c, pU_pc, alpha):
		r""" For the trajectory through the sup-norm space, we automatically compute optimal c
		and advance U along the geodesic

		"""
		M, m = X.shape
		n = self.subspace_dimension
		N = len(self.basis.indices)
		U = orth(U_c[:m*n].reshape(m,n))
		c = U_c[m*n:].reshape(N)
		Delta = pU_pc[:m*n].reshape(m,n)
		pc = pU_pc[m*n:].reshape(N)
	
		#self.set_scale(X, U = U)
		# Orthogonalize	
		Delta = Delta - U.dot(U.T.dot(Delta))

		# Compute the step along the Geodesic	
		Y, s, ZT = scipy.linalg.svd(Delta, full_matrices = False, lapack_driver = 'gesvd')
		U = np.dot(np.dot(U,ZT.T), np.diag(np.cos(s*alpha))) + np.dot(Y, np.diag(np.sin(s*alpha)))

		# Solve a convex problem to actually compute optimal c
		V = self.V(X, U)
		c = inf_norm_fit(V, fX)

		return np.hstack([U.flatten(), c.flatten()])		
			
	def _fit_inf_norm(self, X, fX, U0, **kwargs):
		M, m = X.shape
		n = self.subspace_dimension
		N = len(self.basis.indices)
		
		if U0 is None:
			U0 = self._fit_affine_2_norm(X, fX)
			if self.subspace_dimension > 1:
				U0 = np.hstack([U0, np.random.randn(X.shape[1], self.subspace_dimension-1)])

		# Setup the objective function
		def objfun(U_c, **kwargs):
			return self._inf_residual_grad(X, fX, U_c, **kwargs)
		
		obj = BaseFunction()
		obj.__call__ = objfun
	
		# Trajectory setup
		trajectory = lambda U_c, p, alpha: self._inf_trajectory(X, fX, U_c, p, alpha)

		# Initialize parameter values
		self.set_scale(X, U0)
		V = self.V(X, U0)
		c = inf_norm_fit(V, fX)	
		U_c0 = np.hstack([U0.flatten(), c])

		#TODO: Add orthogonality constraints to search direction

		# Perform optimization
		U_c = minimax(obj, U_c0, trajectory = trajectory, trust_region = True, **kwargs)	
	
		# Store solution	
		self._U = U_c[:m*n].reshape(m,n)
		self.coef = U_c[m*n:].reshape(N)
		

class PolynomialRidgeBound(PolynomialRidgeFunction):
	pass



if __name__ == '__main__':
	np.random.seed(0)
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
	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension  = n, norm = np.inf)
	pra.fit(X, fX)
	#pra._fit_fixed_U_inf_norm(X, fX, U)
	

	# TODO: Fix bug in scaling

#	print pra.U
#	print U
#
#	print pra.coef
#	print prf.coef
#	print pra(X) - prf(X)
	#V =	pra._build_V(X, U)
