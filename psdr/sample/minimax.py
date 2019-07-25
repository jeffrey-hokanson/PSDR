from __future__ import print_function, division

import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist 

from ..domains.domain import DEFAULT_CVXPY_KWARGS

def _cq_center_cvxpy(Y, L, q = 10):
	xhat = cp.Variable(L.shape[1])
	obj = cp.sum([cp.norm(L*xhat - y)**q for y in Y])
	#obj = cp.sum([cp.norm(L*xhat - y) for y in Y])
	prob = cp.Problem(cp.Minimize(obj))
	#prob.solve(**DEFAULT_CVXPY_KWARGS)
	prob.solve()
	return np.array(xhat.value)


def _cq_center(X, L, q = 10, maxiter = 100):
	assert q>= 4, "Require q >= 4"
	Lnorm = np.linalg.norm(L)

	xhat = np.mean(X, axis = 0)

	Y = L.dot(X.T).T

	for it in range(maxiter):
		yhat = L.dot(xhat)
		D = cdist(xhat.reshape(1,-1), Y)
		# D_{q-2}
		d2 = 1./(len(X)*(q-2))*np.sum(dist**(q-2))
		beta = Lnorm**2 * (q-1)*(q-2) * d2

		#u = xhat - 

	return xhat

def minimax_cluster(domain, N, L = None, maxiter = 100, N0 = None, dx_tol = 1e-5, verbose = True):
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
	Xhat = domain.sample(N)

	dx = 0
	for it in range(maxiter):
		Yhat = L.dot(Xhat.T).T
		# Assign each point to its nearest neighbor in L-norm
		D = cdist(Yhat, Y)
		I = np.argmin(D, axis = 0)
		if verbose:
			print("%4d obj: %7.3e; dx %7.3e" % (it, np.max(np.min(D, axis = 0)), dx)) 
		dx = 0
		for i in range(N):
			#Xhat[i] = _cq_center(Y[I == i], L, q = 10)
			xhat = _cq_center_cvxpy(Y[I == i], L, q = 10)
			dx = max(dx, np.linalg.norm(xhat - Xhat[i]))
			Xhat[i] = xhat
		if dx < dx_tol:
			break
	
	if verbose:
		Yhat = L.dot(Xhat.T).T
		D = cdist(Yhat, Y)
		print("%4d obj: %7.3e; dx %7.3e" % (it, np.max(np.min(D, axis = 0)), dx)) 

	return Xhat
