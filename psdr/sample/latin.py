""" Latin Hypercube sampling and variants"""
from __future__ import division, print_function

import numpy as np
from scipy.spatial.distance import squareform, pdist

def _score_maximin(X):
	D = squareform(pdist(X))
	D += np.max(D)*np.eye(D.shape[0])
	min_dist = np.min(D, axis = 0)
	return tuple(np.sort(min_dist))


def latin_hypercube_maximin(domain, N, maxiter = 1000):
	r""" Construct a maximin distance Latin hypercube design

	This function builds a Latin hypercube design on a specified domain, 
	choosing samples that both satisfy the Latin hypercube requirement
	of uniform projection onto the coordinate axes as well as 
	samples that maximize the minimum pairwise distance.  

	

	Parameters
	----------
	domain: Domain
		Domain on which to construct the design
	N: int
		Number of samples to take
	maxiter: int, optional
		Number of iterations to perform trying to find the best design
	"""
	if domain.is_box_domain:
		return _latin_hypercube_maximin_box(domain, N, maxiter = maxiter)
	else:
		from .projection import projection_sample 
		return projection_sample(domain, N, None, maxiter = maxiter, _lhs = True) 


def _latin_hypercube_maximin_box(domain, N, maxiter = 1000):
	r""" Construct a maximin distance Latin hypercube design

	Parameters
	----------
	domain: Domain
		Domain on which to construct the design
	"""

	assert domain.is_box_domain, "This only works on box domains"
	

	# Coordinates along each axis we'll be sampling at
	xs = []
	for i in range(len(domain)):
		xi = np.linspace(domain.norm_lb[i], domain.norm_ub[i], N + 1)
		xi = (xi[1:]+xi[0:-1])/2.
		xs.append(xi)
	
	score_best = tuple(np.zeros(N))
	X_best = None
	# Generate random initial permutations
	perms_best = np.vstack([np.random.permutation(N) for i in range(len(domain))])
	
	for it in range(maxiter):
		# Generate new permutation of the permutations
		i, j = np.random.permutation(N)[0:2]
	
		r = np.random.rand(len(domain))
		I = r >= min(max(r), 0.5)
		perms = np.copy(perms_best)
		perms[I,i] = perms_best[I,j]
		perms[I,j] = perms_best[I,i]

		# Generate samples 
		X = np.array([ [xs[i][j] for j in perms[i]] for i in range(len(domain))]).T

		score = _score_maximin(X)

		if score > score_best:
			perms_best = perms
			X_best = X
			score_best = score

		
	return X_best	

# JMH: I am depreciating this version because it produces far worse designs than the other code

#def latin_hypercube_random(domain, N, metric = 'maximin', maxiter = 100, jiggle = False):
#	r""" Generate a Latin-Hypercube design
#
#	
#
#	This implementation is based on [PyDOE](https://github.com/tisimst/pyDOE/blob/master/pyDOE/doe_lhs.py). 
#
#
#	Parameters
#	----------
#	N: int
#		Number of samples to take
#	metric: ['maximin', 'corr']
#		Metric to use when selecting among multiple Latin Hypercube designs. 
#		One of
#	
#		- 'maximin': Maximize the minimum distance between points, or 
#		- 'corr' : Minimize the correlation between points.
#
#	jiggle: bool, default False
#		If True, randomize the points within the grid specified by the 
#		Latin hypercube design.
#
#	maxiter: int, default: 100
#		Number of random designs to generate in attempting to find the optimal design 
#		with respect to the metric.
#	"""	
#
#	N = int(N)
#	assert N > 0, "Number of samples must be positive"
#	assert metric in ['maximin', 'corr'], "Invalid metric specified"
#
#	xs = []
#	for i in range(len(domain)):
#		xi = np.linspace(domain.norm_lb[i], domain.norm_ub[i], N + 1)
#		xi = (xi[1:]+xi[0:-1])/2.
#		xs.append(xi)
#
#	# Higher score == better
#	score = -np.inf
#	X = None
#	for it in range(maxiter):
#		# Select which components of the hypercube
#		I = [np.random.permutation(N) for i in range(len(domain))]
#		# Generate actual points
#		X_new = np.array([ [xs[i][j] for j in I[i]] for i in range(len(domain))]).T
#
#		# Here we would jiggle points if so desired
#		if jiggle:
#			for i in range(len(domain)):
#				h = xs[i][1] - xs[i][0] # Grid spacing
#				X_new[:,i] += np.random.uniform(-h/2, h/2, size = N)	
#
#		# Evaluate the metric
#		if metric == 'maximin':
#			new_score = np.min(pdist(X_new))
#		elif metric == 'corr':
#			new_score = -np.linalg.norm(np.eye(len(domain)) - np.corrcoef(X_new.T))
#
#		if new_score > score:
#			score = new_score
#			X = X_new
#
#	return X




