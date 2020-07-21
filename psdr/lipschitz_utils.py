import numpy as np
import scipy.linalg
from scipy.optimize import brenth, brentq

def _quad_form(A, X):
	r""" Computes x[i].T @ A @ x[i] efficiently
	"""
	# Looking at https://stackoverflow.com/a/18542236, 
	# advociated a dot product approach
	# i.e., (X.dot(A) * X).sum(axis = 1)
	# However, more recent testing suggests the einsum approach
	# is more efficient
	return np.einsum('ij,jk,ik->i',X, A, X)
	# i.e., (X.dot(A) * X).sum(axis = 1)

def _vec_norm(P):
	r""" Return P[i].T @ P[i]
	"""
	return np.einsum('ij,ji->i',P, P.T)

def dist_eval_constraints(H, X, fX):
	r""" Compute the distance of H to the evaluation constraints

	Parameters
	----------
	H: np.array (m, m)
		Squared Lipschitz matrix
	X: np.array (M, m)
		Location of evaluations
	fX: np.array (M,)
		Value of the function at those locations

	Returns
	-------
	dist: np.array (M*(M-1)//2,)
		Frobenius norm of the perturbation to make a constraint active
		(ordering corresponds to np.triu_indices)
	"""
	M = len(X)

	I, J = np.triu_indices(M, k=1)
	P = (X[I] - X[J])
	pHp = _quad_form(H, P)
	pp2 = _vec_norm(P)

	lhs = (fX[I] - fX[J])**2
	dist = (pHp - lhs)/pp2

	return dist 

def dist_grad_constraints_slow(H, grads):
	r""" Computes the smallest eigenvalue of H - g[k] g[k]'


	
	Parameters
	----------
	H: np.array (m, m)
		Squared Lipschitz matrix
	grads: np.array (N, m)
		Gradients 

	Returns
	-------
	dist: np.array (N)
		Smallest norm of the smallest Frobenius-norm perturbation of H such that 
		(H + delta H) - g[k] g[k]' is indefinite
 
	"""
	N = len(grads)
	dist = np.zeros(N)
	m = H.shape[0]

	ew, U = scipy.linalg.eigh(-H)
	Ugrads = (U.T @ grads.T).T
	eps = np.finfo(np.float64).eps

	# Squared values of entries in Ugrads
	Ugrads2 = Ugrads**2

	# Determine the left and right intervals for root searching
 	
	# This avoids evaluating on the asymptote of the secular equation
	# and pushes slightly to the right (this is confusing because ew[-1] is negative). 
	left = ew[-1]*(1 - eps)

	# the right bound is ew[-1] + || g ||_2^2 
	right = ew[-1] + np.sum(Ugrads2, axis = 1)
		

	for k in range(N):
		def obj(lam):
			# All the terms in the sum will be negative so we don't have
			# catestrophic cancellation
			return 1 + np.sum( Ugrads2[k,:]/(ew - float(lam)))

		# Rough experiments suggest this hyperbolic variant is faster 
		# than the quadratic variant brentq.
		# brenth : 10.25 calls on average
		# brentq : 10.46 calls on average
		# Of course LAPACK with DLAED4 does this in less than 4 function calls;
		# DLAED4 uses a much more sophisticated algorithm, but we use
		# brenth simply as a fall back when the LAPACK version doesn't work 
		
		dist[k] = brenth(obj, left, right[k])

	return -dist


# Use the fast version from LAPACK by default,
# but if impossible, fall back onto the code above
try:
	from .lipschitz_fast import dist_grad_constraints
except ImportError as e:
	import warnings
	# TODO: Reinclude this comment before adding in fast Lipschitz code
	#warnings.warn('Could not load fast gradient code')
	dist_grad_constraints = dist_grad_constraints_slow



def _preprocess_data(X = None, fX = None, grads = None):
	r""" Standardize data for Lipschitz matrix computation
	"""
	if X is not None and len(X) > 0:
		dimension = len(X[0])
	elif grads is not None:
		dimension = len(grads[0])
	else:
		raise Exception("Could not determine dimension of ambient space")
	
	if X is not None and fX is not None:
		N = len(X)
		assert len(fX) == N, "Dimension of input and output does not match"
		X = np.atleast_2d(np.array(X))
		fX = np.array(fX).reshape(X.shape[0])

	elif X is None and fX is None:
		X = np.zeros((0,dimension))
		fX = np.zeros((0,))
	else:
		raise AssertionError("X and fX must both be specified simultaneously or not specified")

	if grads is not None:
		grads = np.array(grads).reshape(-1,dimension)
	else:
		grads = np.zeros((0, dimension))

	return X, fX, grads

def project_evals(H, X, fX):
	r""" Construct a feasible squared Lipschitz matrix given evaluations

	Parameters
	----------
	H: array-like, (m,m)
		Existing lower bound on squared Lipschitz matrix
	X: array-like, (M,m)
		Locations where f is sampled
	fX: array-like (M,)
		Value of f at corresponding evaluations
	"""
	if len(X) == 0:
		return H

	H = np.copy(H) 	# Make sure we don't overwrite existing data
	I, J = np.triu_indices(len(X), k = 1)

	P = (X[I] - X[J])
	lhs = (fX[I] - fX[J])**2
	pp2 = _vec_norm(P)
	while True:
		pHp = _quad_form(H, P)
		update = (lhs - pHp)/pp2**2
		k = np.argmax(update)
		if update[k] > 1e-13:
			H += update[k] * np.outer(P[k], P[k])
		else:
			break
	
	return H

	
def project_grads(H, grads):
	r""" construct a feasible squared Lipschitz matrix H given gradients

	This is done via sequential projection onto the feasible cone 
	
	Parameters
	----------
	H: array-like, (m,m)
		Existing lower bound on squared Lipschitz matrix
	grads: array-like, (M,m)
		Evaluations of the gradient

	Returns
	-------
	H: np.array (m,m)
		Squared Lipschitz matrix satisfying all the gradient constraints 
	"""
	if len(grads) == 0:
		return H

	while True: 
		dist = dist_grad_constraints(H, grads)
		k = np.nanargmin(dist)
		if dist[k] >= 0: break
		g = grads[k]
		ew, V = np.linalg.eigh(H - np.outer(g,g))
		H = np.outer(g, g) + V @ np.diag(np.maximum(0, ew)) @ V.T
	
	return H 



