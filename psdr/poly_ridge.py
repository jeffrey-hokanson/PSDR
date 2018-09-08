"""Ridge function approximation from function values"""
# (c) 2017 Jeffrey M. Hokanson (jeffrey@hokanson.us)

import numpy as np
from itertools import product
import scipy.linalg
from scipy.linalg import norm
from scipy.linalg import svd
from scipy.misc import comb
from scipy.optimize import minimize, check_grad
from scipy.spatial import Voronoi
from copy import deepcopy
from numpy.polynomial.polynomial import polyvander, polyder
from numpy.polynomial.legendre import legvander, legder, legroots 
from numpy.polynomial.chebyshev import chebvander, chebder
from numpy.polynomial.hermite import hermvander, hermder
from numpy.polynomial.laguerre import lagvander, lagder

# Parallel computation
from parallel import pmap
from pgf import PGF
from opt import minimax, linprog 

################################################################################
# Two types of custom errors raised by PolynomialRidgeApproximation
################################################################################
class UnderdeterminedException(Exception):
	pass

class IllposedException(Exception):
	pass


################################################################################
# Linear algebra utility functions
################################################################################

def orth(U):
	""" Orthgonalize, but keep directions"""
	U, R = np.linalg.qr(U, mode = 'reduced')
	U = np.dot(U, np.diag(np.sign(np.diag(R)))) 
	return U


def lstsq(A,b):
	return scipy.linalg.lstsq(A, b, cond = -1)[0]
	#return np.linalg.lstsq(A,b, rcond=-1)[0]

################################################################################
# Indexing utility functions
################################################################################

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

class MultiIndex:
	"""Specifies a multi-index for a polynomial in the monomial basis of fixed total degree 

	"""
	def __init__(self, dimension, degree):
		self.dimension = dimension
		self.degree = degree
		#self.iterator = product(range(0, degree+1), repeat = dimension)	
		self.idx = index_set(degree, dimension)
		self.iterator = iter(self.idx)

	def __iter__(self):
		return self

	def next(self):
		return self.iterator.next()

	def __len__(self):
		return int(comb(self.degree + self.dimension, self.degree, exact = True))

# TODO place this function in a better location,
# and update to use more recent calls
def build_ridge_domain(dom, U):
	if len(U.shape) == 1:
		U = U.reshape(-1,1)
	dim = U.shape[1]

	if dim == 1:
		# One dimensional ridge approximation
		c = U.flatten()
		xp = linprog(c, A_ub = dom.A, b_ub = dom.b, lb = dom.lb, ub = dom.ub, A_eq = dom.A_eq, b_eq = dom.b_eq)
		xn = linprog(-c, A_ub = dom.A, b_ub = dom.b, lb = dom.lb, ub = dom.ub, A_eq = dom.A_eq, b_eq = dom.b_eq)
		ymin = np.dot(c.T, xp)
		ymax = np.dot(c.T, xn)
		ymin, ymax = min(ymin, ymax), max(ymin, ymax)
		zonotope = BoxDomain(ymin, ymax)
	else:
		# In two or more dimensions we resort to Mitchell's best candidate to obtain uniform samples

		# First we constuct an interior approximation of the zonotope
		# Sample points on sphere in dim dimensions
		Z = sample_sphere(dim, 5*10**dim)
		# Construct points on the boundary 
		#inputs = [ ( (dom, U, z), {}) for z in Z]
		#zonotope_points = pmap(extent, inputs, desc = 'zonotope sample')
		zonotope_points = []
		for z in Z:
			x = dom.corner(np.dot(U,z))
			zonotope_points.append(np.dot(U.T, x))
		zonotope_points = np.vstack(zonotope_points)
		# Sample the projected space using Mitchell's best candidate
		zonotope = ConvexHullDomain(zonotope_points) 

	return zonotope


################################################################################
# Defining polynomial bases
################################################################################

class Basis:
	pass

class TensorBasis(Basis):
	""" Generic tensor product basis class, based on momial basis

	This class (and its children) provide the Vandermonde-like matrix
	associated with a multivariate polynomial of total degree p 
	defined in tensor product basis inthe coordinate axes: i.e.,

		f(x) = prod_k \sum_{j=0}^p (x_k)^j.

	
 
	"""
	def __init__(self, n, p):
		self.n = n
		self.p = p
		self.vander = polyvander
		self.der = polyder
		self.indices = index_set(p, n).astype(int)
		self.build_Dmat()

	def build_Dmat(self):
		self.Dmat = np.zeros( (self.p+1, self.p))
		for j in range(self.p + 1):
			ej = np.zeros(self.p + 1)
			ej[j] = 1.
			self.Dmat[j,:] = self.der(ej)

	def V(self, Y):
		""" Builds the Vandermonde matrix associated with this basis

		Given points [Y]_i in R^n, constructs the Vandermonde matrix

			[V]_{i,j} = sum_j prod_k phi_{ [alpha_j]_k} ([Y]_{i,k})

		where alpha_j is an ordering of all multindices with degree less than p

		"""
		M = Y.shape[0]
		V_coordinate = [self.vander(Y[:,k], self.p) for k in range(self.n)]
		
		V = np.ones((M, len(self.indices)), dtype = Y.dtype)
		
		for j, alpha in enumerate(self.indices):
			for k in range(self.n):
				V[:,j] *= V_coordinate[k][:,alpha[k]]
		return V

	def VC(self, Y, C):
		""" Compute the product V(Y) x """
		M = Y.shape[0]
		assert len(self.indices) == C.shape[0]

		if len(C.shape) == 2:
			oneD = False
		else:
			C = C.reshape(-1,1)
			oneD = True

		V_coordinate = [self.vander(Y[:,k], self.p) for k in range(self.n)]
		out = np.zeros((M, C.shape[1]))	
		for j, alpha in enumerate(self.indices):

			# If we have a non-zero coefficient
			if np.max(np.abs(C[j,:])) > 0.:
				col = np.ones(M)
				for ell in range(self.n):
					col *= V_coordinate[ell][:,alpha[ell]]

				for k in range(C.shape[1]):
					out[:,k] += C[j,k]*col
		if oneD:
			out = out.flatten()
		return out

	def DV(self, Y):
		M = Y.shape[0]
		V_coordinate = [self.vander(Y[:,k], self.p) for k in range(self.n)]
		
		mi = MultiIndex(self.n, self.p)
		N = len(mi)
		DV = np.ones((M, N, self.n), dtype = Y.dtype)

		for k in range(self.n):
			for j, alpha in enumerate(MultiIndex(self.n, self.p)):
				for q in range(self.n):
					if q == k:
						DV[:,j,k] *= np.dot(V_coordinate[q][:,0:-1], self.Dmat[alpha[q],:])
					else:
						DV[:,j,k] *= V_coordinate[q][:,alpha[q]]

		return DV


class MonomialTensorBasis(TensorBasis):
	pass

class LegendreTensorBasis(TensorBasis):
	def __init__(self, n, p):
		self.n = n
		self.p = p
		self.vander = legvander
		self.der = legder
		self.indices = index_set(p, n).astype(int)
		self.build_Dmat()

class ChebyshevTensorBasis(TensorBasis):
	def __init__(self, n, p):
		self.n = n
		self.p = p
		self.vander = chebvander
		self.der = chebder
		self.indices = index_set(p, n).astype(int)
		self.build_Dmat()

class LaguerreTensorBasis(TensorBasis):
	def __init__(self, n, p):
		self.n = n
		self.p = p
		self.vander = lagvander
		self.der = lagder
		self.indices = index_set(p,n ).astype(int)
		self.build_Dmat()

class HermiteTensorBasis(TensorBasis):
	def __init__(self, n, p):
		self.n = n
		self.p = p
		self.vander = hermvander
		self.der = hermder
		self.indices = index_set(p, n).astype(int)
		self.build_Dmat()


################################################################################
# Defining the residual and Jacobian for VARPRO-ed objective 
################################################################################

def build_V(U, X, basis, scale = True, UX = None):
	"""
		basis : ['monomial', 'legendre']
			If 'monomial', build V in the monomial basis
	"""

	M, m = X.shape
	if len(U.shape) == 1:
		U = U.reshape(m, -1)
	m, n = U.shape
	
	if UX is not None:
		Y = UX
	else:
		Y = np.dot(U.T, X.T).T
	
	if scale:
		if isinstance(basis, HermiteTensorBasis):
			mean = np.mean(Y, axis = 0)
			std = np.std(Y, axis = 0)
			# In numpy, 'hermite' is physicist Hermite polynomials
			# so we scale by 1/sqrt(2) to convert to the 'statisticians' Hermite 
			# polynomials which are orthogonal with respect to the standard normal
			Y = (Y - mean[None,:])/std[None,:]/np.sqrt(2)
		else:
			lb = np.min(Y, axis = 0)
			ub = np.max(Y, axis = 0)
			Y = 2*(Y-lb[None,:])/(ub[None,:] - lb[None,:]) - 1

	V = basis.V(Y)
	return V

def residual(U, X, fX, basis, **kwargs):
	V = build_V(U, X, basis, **kwargs)
	c = lstsq(V, fX)	
	r = fX - np.dot(V, c)
	return r	


def build_J(U, X, fX, basis, scale = True):
	"""

	Parameters
	----------
	c: np.array
		polynomial coefficients V^+fX
	"""
	M, m = X.shape
	if len(U.shape) == 1:
		U = U.reshape(m, -1)

	m, n = U.shape
	
	Y = np.dot(U.T, X.T).T
	
	if scale:
		if isinstance(basis, HermiteTensorBasis):
			mean = np.mean(Y, axis = 0)
			std = np.std(Y, axis = 0)
			# In numpy, 'hermite' is physicist Hermite polynomials
			# so we scale by 1/sqrt(2) to convert to the 'statisticians' Hermite 
			# polynomials which are orthogonal with respect to the standard normal
			Y = (Y - mean[None,:])/std[None,:]/np.sqrt(2)
			d_scale = 1./std
		else:
			lb = np.min(Y, axis = 0)
			ub = np.max(Y, axis = 0)
			Y = 2*(Y-lb[None,:])/(ub[None,:] - lb[None,:]) - 1
			d_scale = 2./(ub - lb)
	else:
		d_scale = np.ones(n)

	V = basis.V(Y)

	c = lstsq(V, fX)	
	r = fX - np.dot(V, c)

	DV = basis.DV(Y)

	# We precompute the SVD to have access to P_V^perp and V^-
	# via matrix multiplication instead of linear solves 
	Y, s, ZT = svd(V, full_matrices = False) 
	
	N = V.shape[1]
	J1 = np.zeros((M,m,n))
	J2 = np.zeros((N,m,n))

	for ell in range(n):
		for k in range(m):
			DVDU_k = X[:,k,None]*DV[:,:,ell]*d_scale[ell]
			
			# This is the first term in the VARPRO Jacobian minus the projector out fron
			J1[:, k, ell] = np.dot(DVDU_k, c)
			# This is the second term in the VARPRO Jacobian before applying V^-
			J2[:, k, ell] = np.dot((DVDU_k).T, r) 

	# Project against the range of V
	J1 -= np.tensordot(Y, np.tensordot(Y.T, J1, (1,0)), (1,0))
	# Apply V^- by the pseudo inverse
	J2 = np.tensordot(np.diag(1./s),np.tensordot(ZT, J2, (1,0)), (1,0))
	J = -( J1 + np.tensordot(Y, J2, (1,0)))
	return J


# Switch disp to verbose
# OR better: replace this call with 
def grassmann_gauss_newton(U0, X, fX, basis, disp = False, 
	xtol = 1e-7, ftol = 1e-7, gtol = 1e-10, beta = 1e-8, shrink = 0.5, maxiter = 100, reorth = False,
	step0 = 1., history = False, gauss_newton = True, rtol = 0, scale = True):
	""" Ridge function approximation


	Parameters
	----------
	U0: np.ndarray 
		Initial subspace estimate
	X: np.ndarray
		Coordiantes for each sample
	fX: np.ndarray
		Function values
	degree: positive integer
		Degree of polynomial on the transformed coordinates
	disp: boolean
		If true, show convergence history
	xtol: float
		Optimization will stop if the change in U is smaller than xtol
	ftol: float
		Optimization will stop if the change in the objective function is smaller than ftol
	gtol: float
		Optimization will stop if the norm of the gradient is smaller than gtol
	maxiter: int
		Maximum number of optimization iterations
	step0: float
		Initial step length
	shrink: float
		How much to shrink the step length during backtracking line search
	gauss_newton: boolean
		If true, use Gauss-Newton, if false, use gradient descent
	reorth: boolean
		Reorthogonalize things against the subspace U
	history: boolean
		If true, return a third ouput: a dictionary where each key is a list residual, subspace U, gradient, etc.  
	scale: boolean
		If true, scale the projected inputs U^T X onto [-1,1]
	
	Returns
	-------
	"""
	U = np.copy(U0)
	n, m = U.shape
	if m >= 1:
		U = orth(U)

	N, n2 = X.shape
	assert n == n2, "shapes of the subspace and X must match"
	degree = basis.p
	
	if (degree == 1 and m > 1): # "degree 1 polynomial does not permit a subspace of greater than one dimension"
		raise UnderdeterminedException

	if len(MultiIndex(m, degree)) + n*m >= N:
		raise UnderdeterminedException


	V = build_V(U, X, basis, scale = scale) 	# construct the generalized Vandermonde matrix
	c = lstsq(V, fX)				# polynomial coefficients
	r = fX - np.dot(V, c)			# compute the residual
	norm_r = float(norm(r))
	termination_message = 'maxiter exceeded'
	if history:
		hist = {}
		hist['U'] = []
		hist['residual'] = []
		hist['gradient'] = []
		hist['step-length'] = []

	for it in range(maxiter):
		# build the Jacobian
		J = build_J(U, X, fX, basis, scale = scale)

		G = np.tensordot(J, r, (0,0))	# compute the gradient
		if reorth:
			G -= np.dot(U, np.dot(U.T, G))

		if gauss_newton:
			Y, s, ZT = svd(J.reshape(J.shape[0], -1), full_matrices = False, lapack_driver = 'gesvd')
			# Apply the pseudoinverse
			Delta = np.dot(Y[:,:-m**2].T, r)
			Delta = np.dot(np.diag(1/s[:-m**2]), Delta)
			Delta = -np.dot(ZT[:-m**2,:].T, Delta).reshape(U.shape)
			if reorth:
				Delta -= np.dot(U, np.dot(U.T, Delta))
		else:
			Delta = -G

		alpha = np.dot(G.flatten().T, Delta.flatten())
		grad_norm = np.dot(G.flatten().T, G.flatten())
		
		if grad_norm <= gtol:
			t = 0.
			termination_message = "stopped due to small gradient norm"
			break
		
		if alpha >= 0:
			if disp:
				print "Gauss-Newton step not a descent direction"
			Delta = -G
			alpha = -grad_norm
	

		Y, s, ZT = svd(Delta, full_matrices = False, lapack_driver = 'gesvd')
		UZ = np.dot(U, ZT.T)
		t = step0
		maxiter2 = 50
		for it2 in range(maxiter2):
			# Compute new estimate
			U_new = np.dot(UZ, np.diag(np.cos(s*t))) + np.dot(Y, np.diag(np.sin(s*t)))
			# Enforce orthogonality more strictly than the above expression
			U_new = orth(U_new)
			
			# Compute the new residual
			UX_new = np.dot(U_new.T, X.T)
			V_new = build_V(U_new, X, basis, scale = scale)
			c_new = lstsq(V_new, fX)
			r_new = fX - np.dot(V_new, c_new)	
			norm_r_new = float(norm(r_new))
			#print "decrease", norm_r - norm_r_new, norm_r_new/norm_r, "alpha", alpha, "beta", beta, "t", t, "grad %1.4e %1.4e" % (np.max(np.abs(G)),np.min(np.abs(G)))
			if norm_r_new <= norm_r + alpha * beta * t or (norm_r_new < norm_r and (norm_r_new/norm_r) < 0.9): 
				break
			t *= shrink
		

		# Compute distance between U and U_new
		# This will raise an exception if the smallest singular value is greater than one 
		# (hence subspaces numerically equivalent)
		with np.errstate(invalid = 'raise'):
			try:
				dist = np.arccos(svd(np.dot(U_new.T, U), compute_uv = False, overwrite_a = True, lapack_driver = 'gesvd')[-1])
			except FloatingPointError:
				dist = 0.
			

		if it2 == maxiter2-1:
			termination_message = "backtracking line search failed to find a good step"
			break

		# Check convergence criteria
		if (norm_r - norm_r_new)<= ftol:
			if norm_r_new <= norm_r:
				U = U_new
				norm_r = norm_r_new
				c = c_new
			termination_message = "stopped due to small change in residual"
			break

		if norm_r_new <= rtol:
			if norm_r_new <= norm_r:
				U = U_new
				norm_r = norm_r_new
				c = c_new
			termination_message = "stopped due to small residual"
			break
		if dist <= xtol:
			if norm_r_new <= norm_r:
				U = U_new
				norm_r = norm_r_new
				c = c_new
			termination_message = "stopped due to small change in U"
			break

		# copy over values
		U = U_new
		UX = UX_new
		V = V_new
		c = c_new
		r = r_new
		norm_r = norm_r_new
		if history:
			hist['U'].append(U)
			hist['residual'].append(r)
			hist['gradient'].append(G)
			hist['step-length'].append(t)
		if disp:
			print "iter %3d\t |r|: %10.10e\t t: %3.1e\t |g|: %3.1e\t |dU|: %3.1e" %(it, norm_r, t, grad_norm, dist)
	if disp:
		print "iter %3d\t |r|: %10.10e\t t: %3.1e\t |g|: %3.1e\t |dU|: %3.1e" %(it, norm_r_new, t, grad_norm, dist)
		print termination_message

	if history:
		return U, c, norm_r, hist
	else:
		return U, c, norm_r

def build_DVDUc(U, X, c, basis, scale = True):
	""" Build the derivative of V with respect to U


	"""
	M, m = X.shape
	assert U.shape[0] ==  m, "Subspace U has wrong number of rows"
	_, n = U.shape
	N = len(basis.indices)

	Y = np.dot(U.T, X.T).T
	
	if scale:
		if isinstance(basis, HermiteTensorBasis):
			mean = np.mean(Y, axis = 0)
			std = np.std(Y, axis = 0)
			# In numpy, 'hermite' is physicist Hermite polynomials
			# so we scale by 1/sqrt(2) to convert to the 'statisticians' Hermite 
			# polynomials which are orthogonal with respect to the standard normal
			Y = (Y - mean[None,:])/std[None,:]/np.sqrt(2)
			d_scale = 1./std
		else:
			lb = np.min(Y, axis = 0)
			ub = np.max(Y, axis = 0)
			Y = 2*(Y-lb[None,:])/(ub[None,:] - lb[None,:]) - 1
			d_scale = 2./(ub - lb)
	else:
		d_scale = np.ones(n)

	DVDUc = np.zeros((M,m,n))
	DV = basis.DV(Y) 	# Size (M, N, n)
	for k in range(m):
		for ell in range(n):
			DVDUc[:,k,ell] = X[:,k]*np.dot(DV[:,:,ell], c)*d_scale[ell]
	return DVDUc


def max_fixed_ridge(U, X, fX, basis, scale):
	M = X.shape[0]
	N = len(basis.indices)
	m = U.shape[0]
	n = U.shape[1]
	# Solve a linear program to determine the optimal value of c
	V = build_V(U, X, basis, scale = scale)
	# fX - V(U)c <= t
	A_ub1 = np.hstack([-V, -np.ones((M,1))])
	b_ub1 = np.copy(-fX)
	# V(U)c - fX <= t
	A_ub2 = np.hstack([V, -np.ones((M,1))])
	b_ub2 = np.copy(fX)
	A_ub = np.vstack([A_ub1, A_ub2])
	b_ub = np.hstack([b_ub1, b_ub2])
	a = np.zeros(N+1)
	a[-1] = 1
	ct = linprog(a, A_ub = A_ub, b_ub = b_ub)
	c = ct[0:N]
	return c

def max_ridge(U0, X, fX, basis, scale = True, **kwargs):
	"""
		
	Solve the optimization problem
		
		min_U  || fX - V(U) c||_\inf


	"""
	M, m = X.shape
	assert len(fX) == M, "Number of outputs does not match number of inputs"
	assert U0.shape[0] == m, "Subspace dimension does not match input dimension"
	_, n = U0.shape
	
	#basis = LegendreTensorBasis(n, p)
	N = len(basis.indices)

	def mismatch(z, return_gradient = False):
		c = z[0:N]
		U = z[N:].reshape(m,n)

		V = build_V(U, X, basis, scale = scale)
		err = fX - np.dot(V, c)
		if return_gradient is False:
			for i in range(M):
				yield err[i]
			for i in range(M):
				yield -err[i]
		else:
			DVDUc = build_DVDUc(U, X, c, basis, scale = scale)
			for i in range(M):
				gi = np.hstack([-V[i], -DVDUc[i].flatten()])
				yield err[i], gi
			for i in range(M):
				gi = np.hstack([V[i], DVDUc[i].flatten()])
				yield -err[i], gi
	
	def trajectory(z0, dz, t):
		U0 = z0[N:].reshape(m,n)
	
		# Compute the step along the Geodesic	
		Delta = dz[N:].reshape(m,n)
		Y, s, ZT = scipy.linalg.svd(Delta, full_matrices = False, lapack_driver = 'gesvd')
		U = np.dot(np.dot(U0,ZT.T), np.diag(np.cos(s*t))) + np.dot(Y, np.diag(np.sin(s*t)))

		# Solve a linear program to determine the optimal value of c
		V = build_V(U, X, basis, scale = scale)
		# fX - V(U)c <= t
		A_ub1 = np.hstack([-V, -np.ones((M,1))])
		b_ub1 = np.copy(-fX)
		# V(U)c - fX <= t
		A_ub2 = np.hstack([V, -np.ones((M,1))])
		b_ub2 = np.copy(fX)
		A_ub = np.vstack([A_ub1, A_ub2])
		b_ub = np.hstack([b_ub1, b_ub2])
		a = np.zeros(N+1)
		a[-1] = 1
		ct = linprog(a, A_ub = A_ub, b_ub = b_ub)
		c = ct[0:N]
	
		z = np.zeros(N+m*n)
		z[0:N] = c
		z[N:] = U.flatten()
		return z

	z0 = np.zeros(N+m*n)
	z0[N:] = U0.flatten()
	if 'c_armijo' not in kwargs:
		kwargs['c_armijo'] = 1e-3
	z, t = minimax(mismatch, z0, trajectory = trajectory, **kwargs)
	c = z[:N]
	U = z[N:].reshape(m,n)
	return U, c, t



def parallel_grassmann_gauss_newton(U0s, X, fX, basis, parallel = True, desc = 'ridge fitting', progress = False, **kwargs):
	""" Parallel wrapper for the grassmann_gauss_newton code using pmap
	"""

	args = [ (np.copy(U0), X, fX , deepcopy(basis) )  for U0 in U0s]
	return pmap(grassmann_gauss_newton, args = args, kwargs = kwargs, parallel = parallel, desc = desc, progress = progress )
	

class PolynomialRidgeApproximation:
	def __init__(self, degree = None, subspace_dimension = None, n_init = 1, scale = True,
			U_fixed = None, basis = 'legendre', keep_data = True, norm = 2, bound = None, **kwargs):
		""" Fit a polynomial ridge function to provided data
		
		Parameters
		----------
		degree: non-negative integer
			Polynomial degree to be fit

		subspace_dimension: non-negative integer
			The dimension on which the polynomial is defined

		n_init: positive integer
			The number of random initializations to preform 
			Large values (say 50) help find the global optimum since
			finding the ridge approximation involves a non-convex optimization problem

		U_fixed: matrix
			A fixed subspace for building an approximation 

		scale: bool
			If true, use a scaling and shifting strategy appropreate for the polynomial basis

		**kwargs:
			Additional arguments are passed to the optimizer


		Raises
		------
		IllposedException
			Raised if the requisted ridge approximation does not exist,
			e.g., a 2d linear ridge function which is equivalent to a 1d linear ridge function 
		"""

		assert norm in [1,2,np.inf], "Norm must be one of 1, 2, or np.inf"
		self.norm = norm
		assert bound in [None, 'lower', 'upper']

		if isinstance(basis, basestring):
			if basis == 'monomial':
				basis = MonomialTensorBasis(subspace_dimension, degree)
			elif basis == 'legendre':
				basis = LegendreTensorBasis(subspace_dimension, degree)
			elif basis == 'hermite':
				basis = HermiteTensorBasis(subspace_dimension, degree)
			elif basis == 'laguerre':
				basis = LaguerreTensorBasis(subspace_dimension, degree)
		elif isinstance(basis, Basis):
			degree = basis.p
		else:
			raise NotImplementedError('Basis type not understood')
		
		if subspace_dimension is None:
			subspace_dimension = 1

		if subspace_dimension is 1 and degree is None:
			degree = 5
		if subspace_dimension is 0 and degree is None:
			degree = 0
		if degree is 0 and subspace_dimension is None:
			subspace_dimension = 0
		if degree is 1 and subspace_dimension is None:
			subspace_dimension = 1

		if degree is 1 and subspace_dimension != 1:
			raise IllposedException('Affine linear functions intrinsically only have a 1 dimensional subspace')
		if degree is 0 and subspace_dimension > 0:
			raise IllposedException('The constant function does not have a subspace associated with it')
		if subspace_dimension is 0 and degree > 1:
			raise IllposedException('Zero-dimensional subspaces cannot have a polynomial term associated with them')

		self.degree = degree
		self.subspace_dimension = subspace_dimension
		self.kwargs = kwargs
		self.n_init = n_init
		self.basis = basis 
		self.scale = scale
		self.keep_data = keep_data
		if U_fixed is not None:
			self.U_fixed = np.copy(U_fixed)
		else:
			self.U_fixed = None


	def _fit_fixed_U(self, X, y, U):
		self.U = orth(U)
		self._fix_scale(X)	
		if self.norm == 2:
			V = build_V(self.U, X, self.basis,  scale = self.scale)
			self.c = lstsq(V,y)
		elif self.norm == np.inf:
			self.c = max_fixed_ridge(self.U, X, y, self.basis, scale = self.scale)	
		#self._fix_sign(X, y)

	def _fit_constant(self, X, y):
		self.U = np.zeros((X.shape[1], 0))
		self.c = lstsq(build_V(self.U, X, self.basis, scale = self.scale), y)

	def _fit_affine(self, X, y):
		# Solve the linear least squares problem
		XX = np.hstack([X, np.ones((X.shape[0],1))])
		b = lstsq(XX, y)
		U = b[0:-1].reshape(-1,1)
		self._fit_fixed_U(X, y, U)	

	def fit(self, X, y):
		""" Build ridge function approximation
		"""
		
		if self.keep_data:
			self.X = np.copy(X)
			self.y = np.copy(y)
		
		if self.U_fixed is not None:
			# If we have been provided with a fixed U
			self._fit_fixed_U(self, X, y, self.U_fixed)
			return

		elif self.subspace_dimension == 0 and self.degree == 0:
			# Special case of fitting a constant
			self._fit_constant(X, y)
			return

		elif self.degree == 1 and self.subspace_dimension == 1:
			# Special case of fitting an affine fit	
			self._fit_affine(X, y)
			return

		# Now we assume we're building a non-trivial ridge approximation
		# So build a list of U0s to start with 
		U0s = []
		kwargs = deepcopy(self.kwargs)
		if 'U0' in self.kwargs:
			U0s.append(self.kwargs['U0'].copy())
			del kwargs['U0']

		if len(U0s) < self.n_init:
			# If we're going to try multiple subspaces, try the one generated by a linear fit first
			pra = PolynomialRidgeApproximation(degree = 1, subspace_dimension = 1, n_init = 1, scale = False)
			pra.fit(X, y)
			U0 = pra.U
			if self.subspace_dimension > 1:
				U0 = orth(np.hstack([U0, np.random.randn(U0.shape[0], self.subspace_dimension-1)])) 
			U0s.append(U0)

		for it in range(len(U0s), self.n_init):
			U0s.append(orth(np.random.randn(X.shape[1], self.subspace_dimension)))

		self._best_score = np.inf
		self.refine(X, y, U0s = U0s, **kwargs)
	
	def refine(self, X, y, n_init = 1, U0s = None, **kwargs):
		"""Improve the current estimate

		The current estimate is refined either by passing an array of initial U0s
		or specifying a number of iterations
		"""
	 
		if U0s is None:
			U0s = [ orth(np.random.randn(self.X.shape[1], self.subspace_dimension)) for i in range(n_init) ]

		if self.norm == 2:
			Ucrs = [grassmann_gauss_newton(U0, X, y, self.basis, **kwargs) for U0 in U0s] 
		elif self.norm == np.inf:
			Ucrs = [max_ridge(U0, X, y, self.basis, **kwargs) for U0 in U0s] 

		scores = [Ucr[2] for Ucr in Ucrs]
		I = np.argmin(scores)
		if scores[I] < self._best_score:
			U = Ucrs[I][0]
			self._fit_fixed_U(X, y, U)
			self._fix_sign(X, y)
	
	def _fix_sign(self, X, y):
		if self.U.shape[1] == 1:
			# Fix the sign of the ridge approximation
			# TODO: These should be normalized prior to going into predict ridge
			UXmin = np.min(np.dot(self.U.T, X.T))
			UXmax = np.max(np.dot(self.U.T, X.T))
			if self.predict_ridge(UXmin) > self.predict_ridge(UXmax):
				# If slope isn't positive, make it so it is.
				self._fit_fixed_U(X, y, -self.U)
		else:
			# Otherwise, rotate the basis into the principle coordinates of the reduced space
			Y = np.dot(self.U.T, X.T).T
			Y -= np.mean(Y, axis = 0)
			A = np.dot(Y.T, Y)
			ew, V = np.linalg.eigh(A)
			self._fit_fixed_U(X, y, np.dot(self.U, V))
			
			
	def _fix_scale(self, X):
		if self.scale:
			Y = np.dot(self.U.T, X.T).T
			if isinstance(self.basis, HermiteTensorBasis):
				self._mean = np.mean(Y, axis = 0)
				self._std = np.std(Y, axis = 0)
			else:
				self._lb = np.min(Y, axis = 0)
				self._ub = np.max(Y, axis = 0)


	def _build_Y(self, X = None, Y = None):
		if X is not None:
			Ynew = np.dot(self.U.T, X.T).T	
		elif Y is not None:
			Ynew = Y

		if self.scale:
			if isinstance(self.basis, HermiteTensorBasis):
				# In numpy, 'hermite' is physicist Hermite polynomials
				# so we scale by 1/sqrt(2) to convert to the 'statisticians' Hermite 
				# polynomials which are orthogonal with respect to the standard normal
				Ynew = (Ynew - self._mean[None,:])/self._std[None,:]/np.sqrt(2)
			else:
				Ynew = 2*(Ynew-self._lb[None,:])/(self._ub[None,:] - self._lb[None,:]) - 1
		
		return Ynew
	
	def predict(self, Xnew):
		Xnew = np.atleast_2d(Xnew).reshape(-1, self.input_dim)
		Ynew = self._build_Y(X = Xnew)
		V = self.basis.V(Ynew) 
		return np.dot(V, self.c)

	def predict_ridge(self, Ynew):
		"""
		N.B. Ynew should be in ambient coordinates
		"""
		Ynew = np.atleast_2d(Ynew).reshape(-1,self.subspace_dimension)
		# Correct to work in scaled units
		Ynew = self._build_Y(Y = Ynew)
		# We turn off scaling because we have already corrected for it above
		V = self.basis.V(Ynew)
		return np.dot(V, self.c)

	def predict_der(self, Xnew):
		"""Derivative of ridge approximation
		"""
		Xnew = np.atleast_2d(Xnew).reshape(-1, self.input_dim)
		Ynew = self._build_Y(X = Xnew)
		DV = self.basis.DV(Ynew)
		# Compute dervative
		DV = np.tensordot(DV, self.c, (1,0) )	
		# Correct for scaling
		if self.scale and isinstance(self.basis, HermiteTensorBasis):
			Dscale = np.diag( 1./self._std/np.sqrt(2))
			DV = np.dot(DV, Dscale)
		elif self.scale:
			Dscale = np.diag( 2./(self._ub - self._lb))
			DV = np.dot(DV, Dscale)

		return np.dot(self.U, DV)

	def predict_ridge_der(self, Ynew):
		Ynew = np.atleast_2d(Ynew).reshape(-1, self.subspace_dimension)
		Ynew = self._build_Y(Y = Ynew)
		# Compute dervative
		DV = self.basis.DV(Ynew)
		DV = np.tensordot(DV, self.c, (1,0) )	
		# Correct for scaling
		if self.scale and isinstance(self.basis, HermiteTensorBasis):
			Dscale = np.diag( 1./self._std/np.sqrt(2))
			DV = np.dot(DV, Dscale)
		elif self.scale:
			Dscale = np.diag( 2./(self._ub - self._lb))
			DV = np.dot(DV, Dscale)

		return DV


	def roots(self, val = 0, only_real = True, derivative = False):
		""" Compute the roots of the ridge function

		Computes the roots of the ridge function g when
		
			g(y) = val

		Only supports one dimensional ridge functions.

		Parameters
		----------
		val: float, default: 0
			Value the ridge function should intersect
		only_real: bool, default: True
			Return only the real roots of this polynomial
		derivative: bool, default: False
			Compute roots of derivative polynomial
		"""		 
		assert self.subspace_dimension == 1, "Can only compute roots for 1-d ridges"

		c = np.copy(self.c)
		if isinstance(self.basis, LegendreTensorBasis):
			if derivative:
				c = legder(c)
			c[0] -= val

			# Compute the polynomial roots (unstable for high order)
			roots = legroots(c)

			# Convert to unscaled units
			if self.scale:
				roots = self._lb + (roots+1)*(self._ub - self._lb)/2.
			
			if only_real:
				roots = np.real(roots[np.isreal(roots)])
			
			# Check the roots have been successfully computed
			if len(roots) > 0:
				if derivative:
					#print "Error in roots", self.predict_ridge_der(roots) - val
					assert np.linalg.norm(self.predict_ridge_der(roots) - val, np.inf) < 1e-3, "Roots not accurately computed" 
				else:
					#print "Error in roots", self.predict_ridge(roots) - val
					assert np.linalg.norm(self.predict_ridge(roots) - val, np.inf) < 1e-3, "Roots not accurately computed" 
			return roots	
		raise NotImplementedError

	
	def score(self, X = None, y = None, norm = False):
		if X is None and y is None:
			X = self.X
			y = self.y
		if X is None or y is None:
			raise RuntimeError('Please provide both X and y')

		diff = np.linalg.norm(self.predict(X) - y, 2)
		if norm:
			return diff/np.linalg.norm(y,2)
		else:
			return diff

	def plot(self, axes = None, X = None, y = None, domain = None):
		from matplotlib import pyplot as plt
		if X is None or y is None:
			X = self.X
			y = self.y
		
		if axes is None:
			fig, axes = plt.subplots(figsize = (6,6))

		if self.subspace_dimension == 1:
			Y = np.dot(self.U.T, X.T).flatten()
			lb = np.min(Y)
			ub = np.max(Y)
			
			axes.plot(Y, y, 'k.', markersize = 6)
			xx = np.linspace(lb, ub, 100)
			XX = np.array([self.U.flatten()*x for x in xx])
			axes.plot(xx, self.predict(XX), 'r-', linewidth = 2)

			if domain is not None:
				ridge_domain = build_ridge_domain(domain, self.U)
				axes.axvspan(ridge_domain.lb[0], ridge_domain.ub[0], color = 'b', alpha = 0.15)

		elif self.subspace_dimension == 2:
			Y = np.dot(self.U.T, X.T).T
			# Construct grid
			x = np.linspace(np.min(Y[:,0]), np.max(Y[:,0]), 100)	
			y = np.linspace(np.min(Y[:,1]), np.max(Y[:,1]), 100)
			xx, yy = np.meshgrid(x, y)
			# Sample the ridge function
			UXX = np.vstack([xx.flatten(), yy.flatten()])
			XX = np.dot(self.U, UXX).T
			YY = self.predict(XX)
			YY = np.reshape(YY, xx.shape)
			
			axes.contour(xx, yy, YY, 
				levels = np.linspace(np.min(self.y), np.max(self.y), 20), 
				vmin = np.min(self.y), vmax = np.max(self.y),
				linewidths = 0.5)
			
			# Plot points
			axes.scatter(Y[:,0], Y[:,1], c = self.y, s = 3)

			# plot boundary
			if domain is not None:
				ridge_domain = build_ridge_domain(domain, self.U)	
				Y = ridge_domain.X
				for simplex in ridge_domain.hull.simplices:
					axes.plot(Y[simplex,0], Y[simplex,1], 'k-')
		else:
			raise NotImplementedError

		return axes

	def plot_pgf(self, base_name, X = None, y = None, ridge_range = None):
		"""
		Paramters
		---------
		ridge_range: None or np.ndarray(2,)
			Range on which to sample the ridge approximation
		"""
		if X is None or y is None:
			X = self.X
			y = self.y

		if self.subspace_dimension == 1:
			Y = np.dot(self.U.T, X.T).flatten()
			if ridge_range is None:
				lb = np.min(Y)
				ub = np.max(Y)
			else:
				lb, ub = ridge_range
		
			pgf = PGF()
			pgf.add('Ux', Y.flatten())
			pgf.add('fx', y) 
			pgf.write('%s_data.dat' % base_name)
			
			xx = np.linspace(lb, ub, 100)
			XX = np.array([self.U.flatten()*x for x in xx])
			pgf = PGF()
			pgf.add('Ux', xx)
			pgf.add('predict', self.predict(XX))
			pgf.write('%s_fit.dat' % base_name)
		else:
			raise NotImplementedError

	def box_domain(self):
		""" Return the lower and upper bounds on the active domain

		This only depends on the set of points given as input, so we don't extrapolate too much.
		A convex hull provides a tighter description in higher dimensional space. 
		"""
		UX = np.dot(self.U.T, self.X.T).T
		lb = np.min(UX, axis = 0)
		ub = np.max(UX, axis = 0)
		return [lb, ub] 

	@property
	def input_dim(self):
		return self.U.shape[0]


	def save(self, filename):
		""" Save the ridge approximation to the specified filename
		"""
		with open(filename, 'wb') as out:
			kwargs = {}
			kwargs['U'] = self.U
			kwargs['c'] = self.c
			kwargs['degree'] = self.degree
			kwargs['subspace_dimension'] = self.subspace_dimension
			kwargs['scale'] = self.scale

			if isinstance(self.basis, HermiteTensorBasis):
				kwargs['basis'] = 'hermite'
				if self.scale:
					kwargs['mean'] = self._mean
					kwargs['std'] = self._std
			else:
				if isinstance(self.basis, LegendreTensorBasis):
					kwargs['basis'] = 'legendre'
				elif isinstance(self.basis, MonomialTensorBasis):
					kwargs['basis'] = 'monomial'
				elif isinstance(self.basis, LaguerreTensorBasis):
					kwargs['basis'] = 'laguerre'
				else:
					raise NotImplementedError

				if self.scale:
					kwargs['lb'] = self._lb
					kwargs['ub'] = self._ub

			if self.keep_data:
				kwargs['X'] = self.X
				kwargs['y'] = self.y

			np.savez(out, **kwargs)

	def load(self, filename):
		with open(filename, 'rb') as out:
			data = np.load(out)
			self.U = data['U']
			self.c = data['c']
			self.degree = int(data['degree'])
			self.subspace_dimension =int( data['subspace_dimension'])
			if data['basis'] == 'monomial':
				self.basis = MonomialTensorBasis(self.subspace_dimension, self.degree)
			elif data['basis'] == 'legendre':
				self.basis = LegendreTensorBasis(self.subspace_dimension, self.degree)
			elif data['basis'] == 'hermite':
				self.basis = HermiteTensorBasis(self.subspace_dimension, self.degree)
			elif data['basis'] == 'laguerre':
				self.basis = LaguerreTensorBasis(self.subspace_dimension, self.degree)
			else:
				raise NotImplementedError
			
			self.scale = bool(data['scale'])
			if data['basis'] == 'hermite' and self.scale:
				self._mean = data['mean']
				self._std = data['std']
			elif self.scale:
				self._lb = data['lb']
				self._ub = data['ub']

			try:
				self.X = data['X']
				self.y = data['y']
				self.keep_data = True
			except KeyError:
				pass

class PolynomialRidgeBound(PolynomialRidgeApproximation):
	"""

	Parameters
	----------

	weight: bool
		If true, weight the points according to the area of their corresponding
		Voronoi cell when fitting ridge bound.  This is designed to compensate
		for point sets which are sparsely populated in certain regions

	pos_slope: bool
		If true and one-dimensional, enforce that the slope is positive at 
		each sample
	"""
	def __init__(self, bound = 'upper', weight = False, pos_slope = False, special_1d = False, **kwargs):
		assert bound in ['upper','lower'], "Bound must either be upper or lower bound"
		# Call the parent class 
		PolynomialRidgeApproximation.__init__(self, **kwargs)
		
		self.bound = bound
		self.weight = weight
		if pos_slope:
			assert self.subspace_dimension == 1, "pos_slope only support for 1-d ridge functions"
		self.pos_slope = pos_slope
		self.special_1d = special_1d

	def _fit_1d(self, X, y):
		""" Special fit in the case of an affine model
		"""
		assert self.subspace_dimension == 1 and self.degree == 1, "Only implemented for 1d ridges with linear polynomials"
		
		# Solve the linear least squares problem
		XX = np.hstack([X, np.ones((X.shape[0],1))])

		norm_y = np.linalg.norm(y, 2)**2		
		fun = lambda c: 0.5*np.linalg.norm(np.dot(XX, c) - y, 2)**2/norm_y
		jac = lambda c: np.dot(XX.T, np.dot(XX, c) - y)/norm_y

		c0 = lstsq(XX, y)
		
		if self.bound == 'upper':
			constraints = [{'type': 'ineq', 'fun': lambda c: np.dot(XX, c) - y, 'jac': lambda c: XX}]
			c0[-1] = c0[-1] + 1.5*np.min(np.dot(XX, c0) - y)
		elif self.bound == 'lower':
			constraints = [{'type': 'ineq', 'fun': lambda c: y - np.dot(XX, c), 'jac': lambda c: -1*XX}]
			c0[-1] = c0[-1] - 1.5*np.max(np.dot(XX, c0) - y)
		
		res = minimize(fun, c0,
					jac = jac, 
					constraints = constraints,
					tol = 1e-14,
					options = {'disp': True, 'maxiter': 200} 
				)
		U = res.x[0:-1].reshape(-1,1)
		c1 = np.linalg.norm(U)
		self.U = U/c1
		self.c = np.hstack([res.x[-1], c1])  
		self.scale = False

	def fit(self, X, y):
		self.X = np.copy(X)
		self.y = np.copy(y)
		if self.subspace_dimension == 1 and self.degree == 1 and self.special_1d:
			self._fit_1d(X,y)
			return
		else:
			# Call the parent class to fit U
			PolynomialRidgeApproximation.fit(self, X, y)

		Y = np.dot(self.U.T, X.T).T
		
		if self.weight:
			M = np.zeros((X.shape[0],))
			if self.subspace_dimension == 1:
				I = np.argsort(Y.flatten()).flatten()
				Iinv = np.argsort(I)
				Y = Y[I]
				for i in range(1, X.shape[0]-1):
					M[i] = np.sqrt(0.5*(Y[i+1] - Y[i-1])) 
				M[0] = np.sqrt(0.5*(Y[1] - Y[0]))
				M[-1] = np.sqrt(0.5*(Y[-1] - Y[-2]))

				# Permute back to the order in X
				M = M[Iinv]
				M = np.diag(M)
			else:
				raise NotImplementedError
		else:
			M = np.eye(X.shape[0])
		
		# now fit the bound
		VU = build_V(self.U, X, self.basis, scale = self.scale)
		MVU = np.dot(M, VU)
		My = np.dot(M, y)


		fun = lambda c: 0.5*np.linalg.norm(np.dot(MVU, c) - My, 2)**2
		jac = lambda c: np.dot(MVU.T, np.dot(MVU, c) - My)


		constraints = []
		if self.bound == 'upper':
			constraints.append({'type': 'ineq',
					'fun': lambda c: np.dot(VU, c) - y ,
					'jac': lambda c: VU,
					})
		elif self.bound == 'lower':
			constraints.append({'type': 'ineq',
					'fun': lambda c: y - np.dot(VU, c) ,
					'jac': lambda c: -VU,
					})
	
		if self.pos_slope:
			# Get matrix of derivatives
			DVU = self.basis.DV(Y)[:,:,0]
			constraints.append({'type': 'ineq',
					'fun': lambda c: np.dot(DVU, c),
					'jac': lambda c: DVU,
					})

	
		res = minimize(fun, self.c,
					jac = jac, 
					constraints = constraints, 
				)
		self.c = res.x

class Test:
	def __init__(self, X, fX):
		self.X = X
		self.fX = fX

	def run(self):
		U0s = [ orth(np.random.randn(X.shape[1], 2)) for i in range(5) ]
		Ucrs = parallel_grassmann_gauss_newton(U0s, self.X, self.fX, basis, parallel = True, progress = False, desc = 'ridge fitting', **kwargs)


if __name__ == '__main__':
	np.random.seed(1)
	M = 1000
	m = 10
	n = 2
	p = 3
	X = np.random.uniform(-1,1, size = (M,m))
	a = np.random.randn(m)
	b = np.random.randn(m)
	fX = np.dot(a,X.T)**2 + np.dot(b, X.T)**3
	U, _ = np.linalg.qr(np.random.randn(m,n))
	c, U = max_ridge(X, fX, U, p, scale = True, verbose = True, c_armijo = 0.1, alpha0 = 1)
	print U
	print c
