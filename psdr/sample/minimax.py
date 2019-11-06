from __future__ import print_function, division

import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist, pdist, squareform

from ..domains.domain import DEFAULT_CVXPY_KWARGS

from .poisson import poisson_disk_sample
from .sobol import sobol_sequence


def _cq_center_cvxpy(Y, L, q = 10, xhat = None, solver_opts = {'warm_start': True}):
	xhat_value = xhat
		
	xhat = cp.Variable(L.shape[1])
	if xhat_value is not None:
		xhat.value = xhat_value	
	
	# This is the objective we want to solve, but
	# all the reductions make this formulation too 
	# expensive to use.
	# obj = cp.sum([cp.norm(L*xhat - y)**q for y in Y])

	# Instead we formulate the objective using only
	# matrix operations 
	# L @ xhat 
	Lxhat = cp.reshape(xhat.__rmatmul__(L), (L.shape[1],1))
	# outer product so copied over all points
	LXhat = Lxhat.T.__rmatmul__(np.ones( (len(Y),1)))
	# 2-norm error for all points
	norms = cp.sum((LXhat - Y)**2, axis = 1)
	obj = cp.sum(norms**(q/2.))

	prob = cp.Problem(cp.Minimize(obj))
	prob.solve(**solver_opts)
	return np.array(xhat.value)


#def _cq_center_agd(X, L, q = 10, maxiter = 1000, verbose = False, xtol = 1e-7, xhat0 = None):
#	r""" This computes Cq center using accelerated gradient descent
#	See Algorithm 2 in [MJ18]_.
#
#	"""
#
#	assert q>= 4, "Require q >= 4"
#
#	# Initialization of the two variables
#	if xhat0 is None:
#		xhat0 = np.mean(X, axis = 0)
#	z = np.copy(xhat0)
#	u = np.copy(xhat0)
#
#	lam = 1
#
#	Y = L.dot(X.T).T
#
#	# Compute Lipschitz-smooth constant
#	D = squareform(pdist(Y))
#	beta = (q-1)*(q-2)*np.max(np.sum(D**(q-2), axis = 1))/(len(X)*(q-2))
#	Lnorm = np.linalg.norm(L)	 # TODO: Cache this?
#	beta *= Lnorm**2			# This is added b/c beta is proportional to derivative, and LT*L appears in the derivative
#
#	for it in range(maxiter):
#		lam_new = (1 + np.sqrt(1 + 4*lam**2))/2.
#		gam = (1 - lam)/lam_new
#
#		dZ = np.tile(z, (len(X),1)) - X
#		LLdZ = L.T.dot(L.dot(dZ.T)).T
#
#		d = cdist(L.dot(z).reshape(1,-1), Y).T
#
#		grad = np.sum(LLdZ*d**(q-2), axis = 0)/len(X)
#		u_new = z - 1./beta*grad
#		z_new = (1 - gam)*u_new + gam * u
#		dx = np.linalg.norm(z - z_new)
#
#		# update all the variables		
#		z = z_new
#		lam = lam_new
#		u = u_new
#		
#		if verbose:
#			d = cdist((L.dot(z)).reshape(1,-1), Y)
#			obj = 1/(len(X)*q)*np.sum(d**q)
#			print("\t%4d | %7.3e | %7.3e" % (it, obj, dx))
#		if dx < xtol and it > 10:
#			break
#	
#	return z

def minimax_cluster(domain, N, L = None, maxiter = 30, N0 = None, xtol = 1e-5, verbose = True, q = 10, solver_opts = {}):
	r"""Identifies an approximate minimax design using a clustering technique due to Mak and Joseph

	This function implements a clustering based approach for minimax sampling following [MJ18]_.


	References
	----------
	.. [MJ18] Simon Mak and V. Roshan Joseph.
		Minimax and Minimax Projection Designs Using Clustering.
	 	Journal of Computational and Graphical Statistics. 2018, vol 27:1 pp 166-178
		DOI:10.1080/10618600.2017.1302881
	"""

	if N0 is None:
		N0 = min(N*100, int(1e4))
	N0 = int(N0)

	if 'warm_start' not in solver_opts:
		solver_opts['warm_start'] = True

	if L is None:
		L = np.eye(len(domain))

	# Samples from the domain to cluster
	# TODO: Should these be distributed with respect to the L norm?
	# NOTE: In the original paper used Sobol sequence to generate these points
	X = sobol_sequence(domain, N0)	
	#X = domain.sample(N0)
	Y = L.dot(X.T).T

	# Initial cluster centers
	# We use the first N random points so that we don't end up with 
	# empty regions. This also has the consequence of spreading these initial points
	# well as the leading terms of the Sobol' sequence are widely separated 
	Xhat = X[0:N]

	# movement cluster centers
	dx = 0
	I_old = np.zeros(len(X))
	for it in range(maxiter):
		Yhat = L.dot(Xhat.T).T
		# Assign each point to its nearest neighbor in L-norm
		D = cdist(Yhat, Y)
		I = np.argmin(D, axis = 0)

		if verbose:
			if it == 0:
				print('%4s | %9s | %9s |' % ('iter', 'objective', 'movement'))
				print('-----|-----------|-----------|')
			
			print("%4d | %9.3e | %9.3e |" % (it, np.max(np.min(D, axis = 0)), dx)) 

		if np.all(I_old == I):
			if verbose: print("point sets unchanged")
			break
		
		dx = 0
		for i in range(N):
			xhat = _cq_center_cvxpy(Y[I == i], L, q = 10, xhat = Xhat[i], solver_opts = solver_opts)
			#xhat = _cq_center_agd(X[I == i], L, q = q, xhat0 = Xhat[i])
			dx = max(dx, np.linalg.norm(xhat - Xhat[i]))
			Xhat[i] = xhat

		if dx < xtol:
			print('stopped due to small movement')
			break

		I_old = I

	return Xhat

def minimax_covering(domain, r, L = None, **kwargs):
	r"""Approximate a minimax design using a discrete approximation of the domain.

	This function is a convience wrapper around :meth:`psdr.minimax_covering_discrete`:
	it constructs a discrete approximation of the domain using Poisson disk sampling
	and then selects from those samples points such that the maximum distance between
	any point in the domain and the samples is at most :math:`r`.

	*Note* This method will solve an expensive 0-1 linear program and will not scale well
	if more than a few hundred Poisson disk samples are taken

	Parameters
	----------
	domain: Domain
		Domain from which to sample
	r: float
		The target maximum distance between any point in the domain and
		the returned sample set.
	L: array-like (?,m)
		Weighting matrix on the two-norm
	**kwargs: dictionary
		Additional parameters for :meth:`psdr.minimax_covering_discrete`
	
	Returns
	-------
	X: np.array
		Samples from the domain approximating a minimax sampling
	"""
	# This ensures that no point in the domain is more than r/2 away from one of these samples
	X = poisson_disk_sample(domain, r/4., L = L)
	# This then selects 
	I = minimax_covering_discrete(X, r/2., L = L, **kwargs)
	return X[I]


def minimax_covering_discrete(X, r, L = None, **kwargs):
	r""" Constructs a minimax design on discrete domain by solving a covering problem

	This implements an algorithm due to Tan [Tan13]_ which, given a finite number of points,
	
	
	References
	----------
	.. [Tan13] Minimax Designs for Finite Design Regions
		Matthias H. Y. Tan. 
		Technometrics 55:3, 346-358
		https://doi.org/10.1080/00401706.2013.804439
	"""
	X = np.array(X)
	if L is None:
		L = np.eye(X.shape[1])
	# Index set - 1/true if the node is selected
	I = cp.Variable(len(X), boolean = True) 

	# compute the pairwise distance matrix
	D = squareform(pdist(L.dot(X.T).T))
	# we require that every 
	constraints = [ I.__rmatmul__(D < r) >= np.ones(len(X)) ]

	prob = cp.Problem(cp.Minimize(cp.sum(I)), constraints)
	prob.solve(**kwargs)

	# Convert to a boolean array
	I = np.array(I.value > 0.5, dtype = np.bool)
	return I


def minimax_design_1d(domain, N, L = None):
	r"""

	"""
	if L is None:
		assert len(domain) == 1, "If no L matrix specified, need a 1-d domain"
		L = np.array([1])
	else:
		L = np.atleast_2d(L)
		assert L.shape[0] == 1, "must provide a 1 by m Lipschitz matrix"

	c1 = domain.corner(L.flatten())
	c2 = domain.corner(-L.flatten())

	h = 1./N
	avec = h/2 + h*np.arange(N)
	X = np.array([a*c1 + (1-a)*c2 for a in avec]) 
	return X 

def minimax_design(domain, N, L = None, **kwargs):
	r""" High level interface to minimax designs

	The goal of this function is to automatically choose the best
	minimax design algorithm.  Specifically, it uses the exact 1d 
	solution if possible; if not, it defaults to minimax_cluster.	

	
	"""

	try:
		return minimax_design_1d(domain, N, L = L)
	except AssertionError:
		return minimax_cluster(domain, N, L = L, **kwargs)


