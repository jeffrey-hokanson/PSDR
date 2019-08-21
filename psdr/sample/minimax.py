from __future__ import print_function, division

import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist, pdist, squareform

from ..domains.domain import DEFAULT_CVXPY_KWARGS

from .poisson import poisson_disk_sample


def _cq_center_cvxpy(Y, L, q = 10):
	xhat = cp.Variable(L.shape[1])
	obj = cp.sum([cp.norm(L*xhat - y)**q for y in Y])
	#obj = cp.sum([cp.norm(L*xhat - y) for y in Y])
	prob = cp.Problem(cp.Minimize(obj))
	#prob.solve(**DEFAULT_CVXPY_KWARGS)
	prob.solve()
	return np.array(xhat.value)


def _cq_center_agd(X, L, q = 10, maxiter = 1000, verbose = False, xtol = 1e-7, xhat0 = None):
	r""" This computes Cq center using accelerated gradient descent
	See Algorithm 2 in [MJ18]_.

	"""
	assert q>= 4, "Require q >= 4"

	# Initialization of the two variables
	if xhat0 is None:
		xhat0 = np.mean(X, axis = 0)
	z = np.copy(xhat0)
	u = np.copy(xhat0)

	lam = 1

	Y = L.dot(X.T).T

	# Compute Lipschitz-smooth constant
	D = squareform(pdist(Y))
	beta = (q-1)*(q-2)*np.max(np.sum(D**(q-2), axis = 1))/(len(X)*(q-2))
	Lnorm = np.linalg.norm(L)	 # TODO: Cache this?
	beta *= Lnorm**2			# This is added b/c beta is proportional to derivative, and LT*L appears in the derivative

	for it in range(maxiter):
		lam_new = (1 + np.sqrt(1 + 4*lam**2))/2.
		gam = (1 - lam)/lam_new

		dZ = np.tile(z, (len(X),1)) - X
		LLdZ = L.T.dot(L.dot(dZ.T)).T

		d = cdist(L.dot(z).reshape(1,-1), Y).T

		grad = np.sum(LLdZ*d**(q-2), axis = 0)/len(X)
		u_new = z - 1./beta*grad
		z_new = (1 - gam)*u_new + gam * u
		dx = np.linalg.norm(z - z_new)

		# update all the variables		
		z = z_new
		lam = lam_new
		u = u_new
		
		if verbose:
			d = cdist((L.dot(z)).reshape(1,-1), Y)
			obj = 1/(len(X)*q)*np.sum(d**q)
			print("%4d | %7.3e | %7.3e" % (it, obj, dx))
		if dx < xtol and it > 10:
			break
	
	return z

def minimax_cluster(domain, N, L = None, maxiter = 30, N0 = None, xtol = 1e-5, verbose = True, q = 10):
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

	if L is None:
		L = np.eye(len(domain))

	# Samples from the domain to cluster
	# TODO: Should these be distributed with respect to the L norm?
	X = domain.sample(N0)
	Y = L.dot(X.T).T

	# Initial cluster centers
	# We use the first N random points so that we don't end up with 
	# empty regions 
	Xhat = X[0:N]
	#Xhat = domain.sample(N)

	dx = 0
	I_old = np.zeros(len(X))
	for it in range(maxiter):
		Yhat = L.dot(Xhat.T).T
		# Assign each point to its nearest neighbor in L-norm
		D = cdist(Yhat, Y)
		I = np.argmin(D, axis = 0)
		if verbose:
			print("%4d obj: %7.3e; dx %7.3e" % (it, np.max(np.min(D, axis = 0)), dx)) 
		if np.all(I_old == I):
			if verbose: print("point sets unchanged")
		#	break
		
		dx = 0
		for i in range(N):
			#xhat = _cq_center_cvxpy(Y[I == i], L, q = 10)
			#Ii = (I == i)
			#if np.all(I_old[Ii] == I[Ii]):
			#	# If we haven't changed the elements of this set between iterations
			#	xhat = _cq_center_agd(X[Ii], L, q = q, xhat0 = Xhat[i])
			#else:
			#	xhat = _cq_center_agd(X[Ii], L, q = q, xhat0 = Xhat[i])
			xhat = _cq_center_agd(X[I == i], L, q = q, xhat0 = Xhat[i])
			dx = max(dx, np.linalg.norm(xhat - Xhat[i]))
			Xhat[i] = xhat

		if dx < xtol:
			print('stopped due to small movement')
			break

		I_old = I

	return Xhat

def minimax_covering(domain, r, L = None, **kwargs):
	r""" Find an approximate minimax design by solving a covering problem on a discrete approximation of the domain

	This is mainly a utility wrapper around minimax_covering_discrete
	that automatically discretizes the domain
	
	This follows Tan13
	"""
	X = poisson_disk_sample(domain, r/2., L = L)
	I = minimax_covering_discrete(X, r, L = L, **kwargs)
	return X[I]


def minimax_covering_discrete(X, r, L = None, **kwargs):
	r"""
	
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
