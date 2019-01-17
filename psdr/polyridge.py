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

	def profile_grad(self, X):
		r""" gradient of the profile function g
		"""
		DV = self.DV(X)
		# Compute gradient on projected space
		Df = np.tensordot(DV, self.coef, axes = (1,0))
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

def one_norm_fit(A, b):
	r""" solve 1-norm linear optimization problem

	.. math::

		\min_{x} \| \mathbf{a} \mathbf{x} - \mathbf{b}\|_1

	"""
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', PendingDeprecationWarning)
		x = cp.Variable(A.shape[1])
		obj = cp.norm1(x.__rmatmul__(A) - b)
		problem = cp.Problem(cp.Minimize(obj))
		problem.solve()
		return x.value

def two_norm_fit(A,b):
	r""" solve 2-norm linear optimization problem

	.. math::

		\min_{x} \| \mathbf{a} \mathbf{x} - \mathbf{b}\|_2

	"""
	return scipy.linalg.lstsq(A, b)[0]



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
	
	def _fit_affine(self, X, fX):
		r""" Solves the affine 
		"""
		# TODO: There is often a scaling issue 
		XX = np.hstack([X, np.ones((X.shape[0],1))])
		if self.norm == 1:
			b = one_norm_fit(XX, fX)
		elif self.norm == 2:
			b = two_norm_fit(XX, fX)
		elif self.norm == np.inf:
			b = inf_norm_fit(XX, fX)

		U = b[0:-1].reshape(-1,1)
		return U	


	def _init_U(self, X, fX):
		U0 = self._fit_affine(X, fX)
		if self.subspace_dimension > 1:
			# TODO: Find something better than random for the other directions
			U0 = np.hstack([U0, np.random.randn(X.shape[1], self.subspace_dimension-1)])
		return U0

	def _fit_coef(self, X, fX, U):
		self._U = U
		self.set_scale(X)
		V = self.V(X)
		if self.norm == 1:
			self.coef = one_norm_fit(V, fX)
		elif self.norm == 2:
			self.coef = two_norm_fit(V, fX)
		elif self.norm == np.inf:
			self.coef = inf_norm_fit(V, fX)

	def _finish(self, X, fX, U):
		r""" Given final U, rotate and find coefficients
		"""

		# Step 1: Apply active subspaces to the profile function at samples X
		# to rotate onto the most important directions
		if U.shape[1] > 1:
			self._fit_coef(X, fX, U)
			grads = self.profile_grad(X)
			Ur = scipy.linalg.svd(grads.T)[0]
			U = U.dot(Ur)
		
		# Step 2: Flip signs such that average slope is positive in the coordinate directions
		self._fit_coef(X, fX, U)
		grads = self.profile_grad(X)
		U = U.dot(np.diag(np.sign(np.mean(grads, axis = 0))))
		
		# Step 3: final fit	
		self._fit_coef(X, fX, U)
		grads = self.profile_grad(X)

	################################################################################	
	# Two norm functions
	################################################################################	
	
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
	
	def _grassmann_trajectory(self, U_flat, Delta_flat, t):
		Delta = Delta_flat.reshape(-1, self.subspace_dimension)
		U = U_flat.reshape(-1, self.subspace_dimension)
		Y, s, ZT = scipy.linalg.svd(Delta, full_matrices = False, lapack_driver = 'gesvd')
		UZ = np.dot(U, ZT.T)
		U_new = np.dot(UZ, np.diag(np.cos(s*t))) + np.dot(Y, np.diag(np.sin(s*t)))
		U_new = orth(U_new).flatten()
		return U_new
	
	def _fit_2_norm(self, X, fX, U0, **kwargs):
		if U0 is None:
			U0 = self._init_U(X, fX)	
	
		# Setup scaling	
		self.set_scale(X, U = U0)
		
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
			trajectory = self._grassmann_trajectory, gnsolver = gn_solver, **kwargs) 
		
		U = U_flat.reshape(-1, self.subspace_dimension)
		
		self._finish(X, fX, U)	


	def _fit_fixed_U_2_norm(self, X, fX, U):
		self.U = orth(U)
		self.set_scale(X)
		V = self.V(X)
		self.coef = scipy.linalg.lstsq(V, fX)[0].flatten()

	################################################################################	
	# Generic residual and Jacobian
	################################################################################	

	def _residual(self, X, fX, U_c):
		M, m = X.shape
		N = len(self.basis)
		n = self.subspace_dimension
		
		# Extract U and c
		U = U_c[:m*n].reshape(m,n)
		c = U_c[m*n:].reshape(N)
		
		# Construct basis
		V = self.V(X, U)
		res = V.dot(c) - fX
		return res

	def _jacobian(self, X, fX, U_c):
		M, m = X.shape
		N = len(self.basis)
		n = self.subspace_dimension
		
		# Extract U and c
		U = U_c[:m*n].reshape(m,n)
		c = U_c[m*n:].reshape(N)
		
		# Derivative of V with respect to U with c fixed	
		DVDUc = np.zeros((M,m,n))
		DV = self.DV(X, U) 	# Size (M, N, n)
		for k in range(m):
			for ell in range(n):
				DVDUc[:,k,ell] = X[:,k]*np.dot(DV[:,:,ell], c)
		
		# Derivative with respect to linear component
		V = self.V(X, U)

		# Total Jacobian
		jac = np.hstack([DVDUc.reshape(M,-1), V])
		return jac

		
	def _trajectory(self, X, fX, U_c, pU_pc, alpha):
		r""" For the trajectory through the sup-norm space, we automatically compute optimal c
		and advance U along the geodesic

		"""
		M, m = X.shape
		N = len(self.basis)
		n = self.subspace_dimension
		
		# Split components
		U = orth(U_c[:m*n].reshape(m,n))
		c = U_c[m*n:].reshape(N)

		Delta = pU_pc[:m*n].reshape(m,n)
		pc = pU_pc[m*n:].reshape(N)
	
		# Orthogonalize	
		Delta = Delta - U.dot(U.T.dot(Delta))

		# Compute the step along the Geodesic	
		Y, s, ZT = scipy.linalg.svd(Delta, full_matrices = False, lapack_driver = 'gesvd')
		U = np.dot(np.dot(U,ZT.T), np.diag(np.cos(s*alpha))) + np.dot(Y, np.diag(np.sin(s*alpha)))

		# Solve a convex problem to actually compute optimal c
		V = self.V(X, U)
		if self.norm == np.inf:
			c = inf_norm_fit(V, fX)
		elif self.norm == 1:
			c = one_norm_fit(V, fX)
		elif self.norm == 2:
			c = two_norm_fit(V, fX) 

		return np.hstack([U.flatten(), c.flatten()])		
			
	def _fit_inf_norm(self, X, fX, U0, **kwargs):
		M, m = X.shape
		n = self.subspace_dimension
		N = len(self.basis.indices)
		
		if U0 is None:
			U0 = self._init_U(X, fX)
	
		# Define the objective function	
		# We use two copies with flipped signs so that the minimax problem
		# minimizes the maximum (absolute) deviation
		obj = BaseFunction()
		def residual(U_c):
			r = self._residual(X, fX, U_c)
			return np.hstack([r, -r])
		
		obj.eval = residual

		def jacobian(U_c):
			m = X.shape[1]
			n = self.subspace_dimension
			U = U_c[:m*n].reshape(m,n)
			self.set_scale(X, U)
			J = self._jacobian(X, fX, U_c)
			return np.vstack([J, -J])
	
		obj.grad = jacobian	
	
		# Trajectory
		trajectory = lambda U_c, p, alpha: self._trajectory(X, fX, U_c, p, alpha)

		# Initialize parameter values
		self.set_scale(X, U0)
		V = self.V(X, U0)
		c = inf_norm_fit(V, fX)	
		U_c0 = np.hstack([U0.flatten(), c])

		# Add orthogonality constraints to search direction
		# Recall pU.T @ U == 0 is a requirement for Grassmann optimization
		def search_constraints(U_c, pU_pc):
			M, m = X.shape
			N = len(self.basis)
			n = self.subspace_dimension
			U = U_c[:m*n].reshape(m,n)
			constraints = [ pU_pc[k*m:(k+1)*m].__rmatmul__(U.T) == np.zeros(n) for k in range(n)]
			return constraints

		# Perform optimization
		U_c = minimax(obj, U_c0, trajectory = trajectory, trust_region = False,
			search_constraints = search_constraints, **kwargs)	
	
		# Store solution	
		U = U_c[:m*n].reshape(m,n)
		self._finish(X, fX, U)
	

class PolynomialRidgeBound(PolynomialRidgeFunction):
	pass



if __name__ == '__main__':
	import matplotlib.pyplot as plt

	np.random.seed(3)
	p = 3
	m = 4
	n = 1
	M = 100

	U = orth(np.random.randn(m,n))
	coef = np.random.randn(len(LegendreTensorBasis(n,p)))
	prf = PolynomialRidgeFunction(LegendreTensorBasis(n,p), coef, U)

	X = np.random.randn(M,m)
	fX = prf.eval(X)

	U0 = orth(np.random.randn(m,n))
	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension  = n, norm = np.inf)
	pra.fit(X, fX, verbose = True)
	

	pra.shadow_plot(X, fX)
	plt.show()

