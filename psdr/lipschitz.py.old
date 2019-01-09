import numpy as np
import scipy.linalg
from cvxopt import matrix, solvers
from itertools import combinations
from scipy.optimize import bisect
from scipy.spatial.distance import cdist

def scalar_lipschitz(X = None, fX = None, grads = None ):
	""" Computes a scalar Lipschitz bound

	Finds the scalar L such that 

		| f(X[i]) - f(X[j]) | <= L || X[i] - X[j] ||


	Parameters
	----------
	X: np.ndarray (M, m)
		m samples of x
	fX: np.ndarray (M,)
		values of f at x[i]
	grads: np.ndarray(N,m)
		samples of the gradient of f

	Returns
	-------
	float:
		scalar lipschitz constant
	"""
	if X is not None:
		m = X.shape[1]
	elif grads is not None:
		m = grads.shape[1]
	else:
		assert False, "No input provided"
	
	if X is None:
		X = np.zeros((0,m))
		fX = np.zeros(0)
	if grads is None:
		grads = np.zeros((0,m))

	M = X.shape[0]
	assert len(fX) == M, "Length of samples doesn't match length of function values"

	# Actually compute scalar Lipschitz constant 
	if len(fX)> 0:
		L_samp = np.max([ abs(fX[i] - fX[j]) / np.linalg.norm(X[i] - X[j]) for i,j in combinations(range(M), 2)])
	else:
		L_samp = 0

	if len(grads) > 0:
		L_grad = np.max([np.linalg.norm(grad) for grad in grads])
	else:
		L_grad = 0

	L = max(L_grad, L_samp)
	return L		

def check_lipschitz(L, X = None, fX = None, grads = None, verbose = False):
	"""Compute how much the Lipschitz constraints are violated

	Parameters
	----------
	L: np.ndarray(m,m)
		Lipschitz matrix
	X: np.ndarray (M, m)
		m samples of x
	fX: np.ndarray (M,)
		values of f at x[i]
	grads: np.ndarray(N,m)
		samples of the gradient of f

	Returns
	-------
	float:
		value by which constraints are violated; if positive, no-violation

	"""
	if X is not None:
		m = X.shape[1]
	elif grads is not None:
		m = grads.shape[1]
	else:
		assert False, "No input provided"

	if X is None:
		X = np.zeros((0,m))
		fX = np.zeros(0)
	if grads is None:
		grads = np.zeros((0,m))

	gap = np.inf

	bound_ij = list(combinations(range(len(X)), 2))
	for i, j in bound_ij:
		new_gap = np.linalg.norm(np.dot(L, X[i] - X[j])) - np.abs(fX[i] - fX[j])
		if verbose: print "%3d %3d : %5.2e" % (i,j,new_gap)
		gap = min(gap, new_gap)
	
	LL = np.dot(L.T, L)
	for i, g in enumerate(grads):
		G = np.outer(g, g)
		#print scipy.linalg.eigvalsh(LL - G)
		new_gap = np.min(scipy.linalg.eigvalsh(LL - G))
		if verbose: print "%3d : %5.2e" % (i,new_gap)
		gap = min(gap, new_gap)
	return gap

def make_L(W, X = None, fX = None, grads = None):
	""" Convert W = L^T L to L

	Takes the Cholesky factorization of W with a small
	diagonal perturbation to ensure that the L generated
	does not violate the constraints 
	

	Parameters
	----------
	X: np.ndarray (M, m)
		M samples of x
	f: np.ndarray (M,)
		Values of f at X[i]
	grads: np.ndarray(N,m)
		Samples of the gradient of f

	Returns
	-------
	float:
		Scalar Lipschitz constant

	"""
	
	I = np.eye(W.shape[0])

	def chol(x):
		""" Cholesky factor of L with Tychnov regularization
		"""
		R, D, perm = scipy.linalg.ldl(W + x*I, lower = False)
		d = np.copy(np.diag(D))
		d[d < 0] = 0
		L = np.dot(np.diag(np.sqrt(d)), R.T)
		return L	

	# See if we need any reguralization	
	L = chol(0.)
	if check_lipschitz(L, X = X, fX = fX, grads = grads) >= 0:
		return L

	# If we do, determine a crude bound on how far to go
	eps = 1e-14
	while check_lipschitz(chol(eps), X = X, fX = fX, grads = grads) < 0:
		eps *= 10

	# Now root find for optimal value

	obj = lambda x:  check_lipschitz(chol(x), X = X, fX = fX, grads = grads)
	
	# Bisect search to find tolerance
	eps = bisect(obj, 0, eps, xtol = 1e-14)

	# Return L
	L = chol(eps)
	return L
	


def multivariate_lipschitz(X = None, fX = None, grads = None, U = None, verbose = False,
	maxiter = 100, abstol = 1e-7, reltol = 1e-6, feastol = 1e-7, refinement = 1):
	""" Construct a Lipschitz matrix from a set of samples and/or gradients


	Given M samples at points X and values fX as well as N gradients, solve the 
	semidefinite program

		minimize_{W}  || W ||_fro^2
		such that     |fX[i] - fX[j] |^2 <= (X[i] - X[j])^T W (X[i] - X[j])  forall 1<=i<j < M
		              grads[k] grads[k]^T <= W      forall 1<= k <= N
		              W = W^T, 0 <= W

	This implementation uses CVXOPT semidefinite programming (cvxopt.solvers.sdp)

	Parameters
	----------
	X: np.ndarray(M,m)
		Sample locations
	fX: np.ndarray(M,)
		Evaluation of f at samples in X; fX[i] = f(X[i])
	grads: np.ndarray (N,m)
		Samples of gradient
	U: np.ndarray(m,m)
		[Optional] basis for parameterizing W
	verbose: bool, [optional] 	
		If True, provide convergence history
	maxiter: int
		Maximum number of interations for CVXOPT
	abstol: float
		absolute accuracy for CVXOPT
	reltol: float
		relative accuracy for CVXOPT
	feastol: float
		tolerance for feasability conditions for CVXOPT
	refinement: int
		number of steps of iterative refinement for CVXOPT

	Returns
	-------
	np.ndarray (m,m)
		Multivariate Lipschitz matrix
	"""
	assert (X is not None and fX is not None) or (grads is not None), "Invalid input"
	if (X is not None) or (fX is not None):
		assert (X is not None) and (fX is not None), "Both X and fX must be provided"

	if X is not None:
		m = X.shape[1]
		if grads is not None:
			assert grads.shape[1] == m, "Gradients not the same dimension as input"
	elif grads is not None:
		m = grads.shape[1]

	if X is None:
		X = np.zeros((0,m))
		fX = np.zeros(0)
	if grads is None:
		grads = np.zeros((0,m))

	# Normalize the input/output pairs	
	if len(X) > 0:	
		f_min = np.min(fX)
		f_max = np.max(fX)
		scale = (f_max - f_min)
	else:
		f_min = 0
		f_max = 1
		#scale = np.max([np.linalg.norm(grads) for grad in grads])
		scale = 1.

	fX_norm = (fX - f_min)/scale
	grads_norm = grads/scale
	
	# Coordinates for basis of space
	if U is None:
		U = np.eye(m)
	
	# Build basis for symmetric matrices
	Es = []
	for i in range(U.shape[1]):
		for j in range(i+1):
			E = np.outer(U[:,i] + U[:,j], U[:,i] + U[:,j])
			if i == j:
				Es.append(E/4)
			else:
				Es.append(E/2)


	# Constraint matrices for CVXOPT
	Gs = []
	hs = []

	# Append constraints from samples
	for i, (xi, fi) in enumerate(zip(X, fX_norm)):
		for xj, fj in zip(X[:i], fX_norm[:i]):
			hij = xi - xj
			hij_norm = np.linalg.norm(hij)
			vij = hij/hij_norm
			G = np.vstack([-np.dot(vij, np.dot(E, vij)) for E in Es]).T
			Gs.append(matrix(G))
			hs.append( matrix([[ -(fi - fj)**2/hij_norm**2]]))

	
	# Flatten G in column-major (Fortran) ordering for CVXOPT
	G = np.vstack([-E.flatten('F') for E in Es]).T
	#G = np.vstack([-E[np.tril_indices(m)] for E in Es]).T
	G = matrix(G)	
	
	# Add constraint to enforce this matrix is positive-semidefinite
	Gs.append(G)
	hs.append(matrix(np.zeros((m,m))))

	# Build constraints 	
	if grads is not None:
		for grad in grads_norm:
			Gs.append(G)
			H = -np.outer(grad, grad)
			hs.append(matrix(H))

	# Setup objective	
	c = np.array([ np.trace(E) for E in Es])
	c = matrix(c)


	solvers.options['show_progress'] = verbose
	solvers.options['maxiters'] = maxiter
	solvers.options['abstol'] = abstol
	solvers.options['reltol'] = reltol
	solvers.options['feastol'] = feastol
	solvers.options['refinement'] = refinement
	sol = solvers.sdp(c, Gs = Gs, hs = hs)

	z = sol['x']
	W = np.zeros((m,m))
	for Ei, zi in zip(Es, z):
		W += zi*Ei
	
	W *= scale**2
	L = make_L(W, X = X, fX = fX, grads = grads)	
	return L



if __name__ == '__main__':
	from lipschitz_ip import multivariate_lipschitz_ip
	M = 50
	m = 5
	np.random.seed(0)
	A = np.random.randn(m,2)
	A = np.dot(A, A.T)
	print np.linalg.svd(A, compute_uv = False)
	f = lambda x: 0.5*np.dot(x, np.dot(A, x))
	grad = lambda x: np.dot(A, x)
	if False:
		a = np.zeros(m)
		a[0] = 1
		f = lambda x: np.dot(a, x)
		grad = lambda x: a

	X = np.random.uniform(-1,1, size = (M,m))
	fX = np.hstack([f(x) for x in X])
	grads = np.array([grad(x) for x in X])

	
	L = multivariate_lipschitz(grads = grads)
	print L
	print check_lipschitz(L, X = X, fX = fX, grads = grads)
	
	#W = multivariate_lipschitz_sdp(grads = grads)
	
