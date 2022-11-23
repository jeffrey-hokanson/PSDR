from __future__ import print_function, division

import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import linear_sum_assignment
import scipy.special
from functools import lru_cache
from iterprinter import IterationPrinter

from ..domains.domain import DEFAULT_CVXPY_KWARGS

from .poisson import poisson_disk_sample
from .sobol import sobol_sequence

def _align_vector(X, Xhat, L = None):
	r""" Returns Xhat aligned to the order of X
	"""
	if L is None:
		Y = X
		Yhat = Xhat
	else:
		Y = (L @ X.T).T
		Yhat = (L @ Xhat.T).T
	
	D = cdist(Y, Yhat)
	row, col = linear_sum_assignment(D)
	return Xhat[col[np.argsort(row)]]

def _int_bisect(fun, n1, n2, verbose = True):
	r""" Perform bisection over the integers
	
	Returns the smallest integer between n1 and n2 where fun is positive
	"""
	assert n1 < n2, "n1 must be less than n2"

	n1 = int(n1)
	n2 = int(n2)
	f1 = fun(n1)
	f2 = fun(n2)
	assert f1*f2 < 0, "There must be a sign change between the two limits"

	it = 0
	if verbose:
		printer = IterationPrinter(it = '4d', left = '6d', right = '6d', f1 = '9.2e', f2 = '9.2e')
		printer.print_header(it = 'iter', left = 'left', right = 'right', f1 = 'f left', f2 = 'f right')
		printer.print_iter(it = it, left = n1, right =n2, f1 = f1, f2 = f2)

	while True:
		it += 1
		n3 = (n1 + n2)//2
		# If we collide with left boundary, round up
		if n3 == n1:
			n3 = n3 + 1
		if n3 == n2:
			break

		f3 = fun(n3)
		if f3*f1 > 0:
			n1, f1 = n3, f3
		else:
			n2, f2 = n3, f3
		
		if verbose:
			printer.print_iter(it = it, left = n1, right =n2, f1 = f1, f2 = f2)
	

	if f1 > 0:
		return n1
	else:
		return n2

def _cq_center_cvxpy(Y, L, q = 10, xhat = None, solver_opts = {'warm_start': True}, domain = None):
	xhat_value = xhat
		
	xhat = cp.Variable(L.shape[1])
	if xhat_value is not None:
		xhat.value = np.copy(xhat_value)	
	
	# This is the objective we want to solve, but
	# all the reductions make this formulation too 
	# expensive to use.
	# obj = cp.sum([cp.norm(L*xhat - y)**q for y in Y])

	# Instead we formulate the objective using only
	# matrix operations 
	# L @ xhat 
	#Lxhat = cp.reshape(xhat.__rmatmul__(L), (L.shape[0],1))
	# outer product so copied over all points
	#LXhat = Lxhat.T.__rmatmul__(np.ones( (len(Y),1)))
	# 2-norm error for all points
	ones_vec = np.ones((Y.shape[0],1))
	obj = cp.mixed_norm(ones_vec @ cp.reshape(L @ xhat,(1,L.shape[0])) - Y, 2, q)
	#norms = cp.sum((LXhat - Y)**2, axis = 1)
	#obj = cp.sum(norms**(q/2.)
	constraints = []
	if domain is not None:
		constraints += domain._build_constraints(xhat)

	prob = cp.Problem(cp.Minimize(obj), constraints)
	prob.solve(**solver_opts)
	return np.array(xhat.value)


def minimax_optimal_cover(domain, r, L = None, X = None, **kwargs):
	r""" Compute a minimax clustering with a specified max distance

	This is an expensive function, computing coverings until one with the desired radius is found
	"""
	
	if 'verbose' not in kwargs:
		kwargs['verbose'] = False

	if X is None:
		X = sobol_sequence(domain, 1e4)

	@lru_cache(maxsize = None)
	def design(N):
		return minimax_cluster(domain, N, L = L, X = X, **kwargs)
	
	Y = (L.T @ X.T).T
	def maximin_distance(N):
		Xhat = design(N)
		Yhat = (L.T @ Xhat.T).T
		return np.max(np.min(cdist(Yhat, Y), axis = 0))
		
	domain_volume = domain.volume()
	ball_volume = np.pi**(len(domain)/2)/scipy.special.gamma(len(domain)/2 + 1)

	# These are the lower and upper bounds on the covering number
	N0 = int(np.floor((1/r)**len(domain) * (domain_volume/ball_volume) * np.abs(np.linalg.det(L))))
	N1 = int(np.floor((3/r)**len(domain) * (domain_volume/ball_volume) * np.abs(np.linalg.det(L))))
	
	# Do a simple bisection search
	N = _int_bisect(lambda N: r - maximin_distance(N), N0, N1)
	return design(N)


def minimax_cluster_pso(domain, N, L = None, maxiter = 10, X = None, N0 = None, verbose = True,
	n_particles = 5, a1 = 10, a2 = 1, a3 = 1, a4 = 0 , remove_worst = True, q = 10,
	solver_opts = {} ):
	r""" Minimax optimal designs produced using clustering and partical swarm optimization

	This is reminicent of Algorithm 3 in [MJ18]_.

	In the original algorithm, a momentum style approach was used
	to update each location based on differences from the global best design
	and the best design from the current particle. The problem is with a 
	poor initialization, this leads to nodes flying outside of the domain.

	Instead, we update each node by taking convex combinations 
	of several terms with random weightings

		* the recommended new node as per clustering
		* the global best design
		* the local best design
		* random points inside the domain 

	The difference being rather than using a momentum based approach 
	for computing the "velocity" associated with each point.

	
	"""
	if L is None:
		L = np.eye(len(domain))
	
	# Initialize each particle
	# TODO: Do this using "scrambled Sobol' sequences
	#Xhats = [np.copy(domain.sample(N)) for i in range(n_particles)]
	def random_init():
		Xhat = np.zeros((N, len(domain)))
		Xhat[0] = domain.sample(1)
		for j in range(1,N):
			Xt = domain.sample(10*N)
			k = np.argmax(np.min(cdist( (L @ Xhat[:j].T).T, (L @ Xt.T).T)))
			Xhat[j] = Xt[k]
		return Xhat
	
	Xhats = [random_init() for i in range(n_particles)]

	#Xhats[0] = sobol_sequence(domain, N) 
	Xhats_local = [np.copy(X) for X in Xhats]
	Xhat_global = np.copy(Xhats[0])
	best_scores_local = np.inf*np.ones(n_particles)
	best_score_global = np.inf

	Vs = [0*Xhat for Xhat in Xhats]


	# Construct samples of the domain for clustering purposes
	if N0 is None:
		N0 = min(N*100, int(1e4))
	
	N0 = int(N0)

	if X is None:
		X = sobol_sequence(domain, N0)	
	
	Y = (L @ X.T).T

	if verbose:
		printer = IterationPrinter(it = '4d', k = '3d', global_score = '20.10e', local_score = '9.2e', diff_global = '9.2e') 
		printer.print_header(it = 'iter', k = 'particle', global_score = 'global score', local_score = 'local score', diff_global = 'diff global')

	# The first loop 
	for it in range(maxiter):
		for k in range(n_particles):
			# STEP 1: Perform one step updating the clustering
			Xhat = Xhats[k]
			Xhat_local = Xhats_local[k]
			
			# Assign each point to its nearest neighbor in L-norm
			Yhat = (L @ Xhat.T).T
			D = cdist(Yhat, Y)
			I = np.argmin(D, axis = 0)

			# Find new location via clustering
			Xhat_new = np.copy(Xhat)
			for i in range(N):
				try:
					Xhat_new[i,:] = _cq_center_cvxpy(Y[I == i], L, q = q, xhat = Xhat[i], solver_opts = solver_opts, domain = domain)
				except ValueError:
					print("value error")
					pass
			

			# Construct random convex combination
			A1 = a1*np.random.uniform(size = Xhat_new.shape)
			A2 = a2*np.random.uniform(size = Xhat_new.shape)
			A3 = a3*np.random.uniform(size = Xhat_new.shape)
			A4 = a4*np.random.uniform(size = Xhat_new.shape)
				

			A_sum = A1 + A2 + A3 + A4
			A1 /= A_sum
			A2 /= A_sum
			A3 /= A_sum
			A4 /= A_sum

			Z = domain.sample(N)
			Xhat_new = (A1 * _align_vector(Xhat, Xhat_new) + 
				A2 * _align_vector(Xhat, Xhat_local) + 
				A3 * _align_vector(Xhat, Xhat_global) + 
				A4 * Z
				)

			move = np.linalg.norm(Xhat_new - Xhat, 'fro')
			Xhats[k] = Xhat_new
			
			# Score this new design 
			Yhat = (L @ Xhats[k].T).T
			dist = np.max(np.min(cdist(Yhat, Y), axis = 0))
			if dist < best_scores_local[k]:
				best_scores_local[k] = dist
				Xhats_local[k] = np.copy(Xhats[k])

			if dist < best_score_global:
				best_score_global = dist
				Xhat_global = np.copy(Xhats[k])

			diff_global = max([np.linalg.norm(Xhat_global - _align_vector(Xhat_global, Xhats[i]), 'fro') for i in range(n_particles)])
			if verbose:
				printer.print_iter(it = it, k = k, global_score = best_score_global, local_score = best_scores_local[k], diff_global = diff_global)
		
		# For the worst 
		if remove_worst:
			k = np.argmax(best_scores_local)
			Xhats[k] = random_init() 	
		
	return Xhat_global, best_score_global	
			
	


def minimax_cluster(domain, N, L = None, maxiter = 50, N0 = None, xtol = 1e-5, 
	verbose = True, q = 10, solver_opts = {}, X = None, Xhat = None):
	r"""Identifies an approximate minimax design using a clustering technique due to Mak and Joseph

	This function implements a clustering based approach for minimax sampling following [MJ18]_.
	We do not implement the particle swarm optimization here; only the clustering approach.
	Futher, we do not use their recommended gradient descent approach to find the C_q cluster centers,
	instead relying on CVXPY (and by default, ECOS) to solve this problem efficiently and accurately.


	References
	----------
	.. [MJ18] Simon Mak and V. Roshan Joseph.
		Minimax and Minimax Projection Designs Using Clustering.
	 	Journal of Computational and Graphical Statistics. 2018, vol 27:1 pp 166-178
		DOI:10.1080/10618600.2017.1302881

	Parameters
	----------
	domain: Domain
		Domain on which to construct the design
	N: int
		Number of points in the design
	L: None or array-like
		If specified, the weighted 2-norm metric for distance on this space
	maxiter: int
		Maximum number of clustering iterations to pursue
	xtol: float
		Smallest movement in cluster centers before iteration stops
	verbose: bool
		If true, print convergence information
	q: positive float
		Power to raise the 2-norm to, such that we better approximate sup-norm
	solver_opts: dict
		Additional arguments to pass to cvxpy when solving each step	
	X: array-like (M,m)
		Discretization of the domain to use for clustering
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
	if X is None:
		X = sobol_sequence(domain, N0)	
		# Initial cluster centers
		# We use the first N random points so that we don't end up with 
		# empty regions. This also has the consequence of spreading these initial points
		# well as the leading terms of the Sobol' sequence are widely separated
		if Xhat is None: 
			Xhat = X[0:N]
	else:
		if Xhat is None:
			# For the same reason as above, we choose Xhat 
			# from the nearest points to the Sobol sequence
			Xhat = sobol_sequence(domain, N)
			D = cdist(Xhat, X)
			_, col = linear_sum_assignment(D)
			Xhat = X[col]

	Y = L.dot(X.T).T

	# movement cluster centers
	dx = np.nan
	I_old = np.zeros(len(X))

	if verbose:
		printer = IterationPrinter(it = '4d', obj = '12.6e', move = '9.3e')
		printer.print_header(it = 'iter', obj = 'max_x min_j dist', move = 'movement')

	best_Xhat = np.copy(Xhat)
	best_dist = np.inf

	for it in range(maxiter):
		Yhat = L.dot(Xhat.T).T
		# Assign each point to its nearest neighbor in L-norm
		D = cdist(Yhat, Y)
		I = np.argmin(D, axis = 0)
		dist = np.max(np.min(D, axis = 0))

		if dist < best_dist:
			best_Xhat = np.copy(Xhat)
			best_dist = float(dist)

		if verbose:
			printer.print_iter(it = it, obj = np.max(np.min(D, axis = 0)), move = dx) 

		if np.all(I_old == I):
			if verbose: print("point sets unchanged")
			break

		# TODO: This is easy to parallelize, but attempts have not improved wall clock time
		Xhat_new = [ _cq_center_cvxpy(Y[I == i], L, q = q, xhat = Xhat[i], solver_opts = solver_opts) for i in range(N)]
		Xhat_new = np.array(Xhat_new)
		dx = np.max(Xhat_new - Xhat)	
		Xhat = Xhat_new

		if dx < xtol:
			if verbose: print('stopped due to small movement')
			break

		I_old = I

	return best_Xhat

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


