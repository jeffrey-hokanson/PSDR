""" Utility containing various geometric routines, most of which are used in sampling
"""
import numpy as np
from scipy.spatial import Voronoi 


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


def voronoi_vertices(X, domain = None, check_samples = False):
	""" Compute all the Voronoi vertices  

	This function provides a uniform access to the Voronoi vertices,
	those points in n-dimensional space that are equidistant from n+1 points.
	This is necessary since QHull, and consequently scipy.spartial.Voronoi
	only can handle 2-dimensional and higher.

	If a domain is provided, this function restricts vertices to that domain
	and also returns intersection of the Voronoi edges with the boundary. 

	Parameters
	----------
	X: np.ndarray(M, n)
		Array of points 
	domain: None or Domain 
		Domain on which to restrict 	


	Returns
	-------
	np.ndarray(N, n)
		points satisfying (n+1) equality constraints 
		from both the distances between points and the domain
	"""
	if len(X.shape) == 1:
		X = X.reshape(-1,1)

	if domain is not None:
		assert len(domain) == X.shape[1], "Dimension of domain doesn't match the samples"
		if check_samples:
			assert np.all(domain.isinside(X)), "Not all points inside domain"

	if len(X.shape) == 1 or X.shape[1] == 1:
		# Q-hull doesn't handle the 1d case because it is straightfoward
		yhat = np.sort(Y.flatten())
		vertices = 0.5*(yhat[1:] + yhat[:-1]).reshape(-1,1)
		if domain is not None:
			# If we have a domain, also include the boundaries
			c1 = domain.corner(np.ones(1))		
			c2 = domain.corner(-np.ones(1))	
			c1, c2 = min(c1, c2), max(c1,c2)
			vertices = np.hstack([c1, vor, c2])
		vertices = vertices.reshape(-1,1)
	else:
		vor = Voronoi(X)
		if domain is None:
			vertices = vor.vertices
		else:
			I = domain.isinside(vor.vertices)
			print I
			vertices = vor.vertices[I]
			print vor.ridge_vertices	
	return vertices 


if __name__ == '__main__':
	from domains import BoxDomain
	dom = BoxDomain([-1,-1,-1],[1,1,1])
	X = dom.sample(10)
	vert = voronoi_vertices(X, domain = dom)	
	print vert
