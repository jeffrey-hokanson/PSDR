""" Utility containing various geometric routines, most of which are used in sampling
"""
from __future__ import print_function
import numpy as np
from scipy.spatial import Voronoi 
from scipy.spatial.distance import cdist, pdist, squareform
from .domains import EmptyDomain

def sample_sphere(dim, n, k = 100):
	""" Sample points on a high-dimensional sphere 

	Uses Mitchell's best candidate algorithm to obtain a 
	quasi-uniform distribution of points on the sphere,


	See:
		https://www.jasondavies.com/maps/random-points/
		https://bl.ocks.org/mbostock/d7bf3bd67d00ed79695b

	Parameters
	----------
	dim: int, positive
		Dimension of the space to sample
	n: int, positive
		Number of points to sample
	k: int, positive (optional)
		Number of candidates to take at each step
	"""
	X = np.zeros( (n, dim) )
	
	# First sample
	x = np.random.randn(dim)
	x /= np.linalg.norm(x)
	X[0] = x
	
	for i in range(1,n):
		# Draw candidates (normalized points on the sphere)
		Xcan = np.random.randn(k, dim)
		Xcan = (Xcan.T/np.sqrt(np.sum(Xcan**2, axis = 1))).T

		# Compute the distance
		dist = np.min(1 - np.dot(X[:i,], Xcan.T), axis = 0)
		I = np.argmax(dist)
		X[i] = Xcan[I]

	return X


