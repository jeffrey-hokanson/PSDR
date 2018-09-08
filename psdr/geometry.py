""" Utility containing various geometric routines, most of which are used in sampling
"""
import numpy as np
from scipy.spatial import Voronoi 
from scipy.spatial.distance import cdist
from domains import EmptyDomain

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


def voronoi_vertices(X):
	""" Compute all the Voronoi vertices  

	This function provides a uniform access to the Voronoi vertices,
	those points in n-dimensional space that are equidistant from n+1 points.
	This is necessary since QHull, and consequently scipy.spartial.Voronoi
	only can handle 2-dimensional and higher.


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

	if len(X.shape) == 1 or X.shape[1] == 1:
		# Q-hull doesn't handle the 1d case because it is straightfoward
		xhat = np.sort(X.flatten())
		vertices = 0.5*(xhat[1:] + xhat[:-1]).reshape(-1,1)
		vertices = vertices.reshape(-1,1)
	else:
		vor = Voronoi(X)
		vertices = vor.vertices
	return vertices 

def candidate_furthest_points(X, domain, L = None, nboundary = 100, n_samp = 10):
	""" Generate points which have the potential to be the furthest from others in the domain 
	"""
	if L is None:
		Y = np.copy(X)
	else:
		Y = np.dot(L, X.T).T
		U, s, VT = np.linalg.svd(L)
		
	# First we construct samples on the interior of the domain
	Yinterior = voronoi_vertices(Y)
	if L is None:
		# Restrict to those samples inside the domain
		Xinterior = Yinterior[domain.isinside(Yinterior)]
	elif Y.shape[1] < X.shape[1]:
		# If there is dimension reduction, 
		Xinterior = []
		for yint in Yinterior:
			try:
				# To find the corresponding point in the untransformed space, 
				# setup an equality constrainted problem
				dom_eq = domain.add_constraint(A_eq = L, b_eq = yint)

				# Randomly sample the unconstrained dimensions
				Xcan = dom_eq.sample(n_samp)

				# Pick the sample furthest from existing samples
				dist = np.min(cdist(Xcan, X), axis = 1)
				k = np.argmax(dist)
				Xinterior.append(Xcan[k])	
			
			except EmptyDomain, InfeasibleConstraints:
				pass
		Xinterior = np.array(Xinterior)
	else:
		# Without dimension reduction, we simply compute the inverse to 
		# map samples of L*x back to x.
		Xinterior = VT.T.dot(np.diag(1./s).dot(U.T.dot(Yinterior.T))).T
		I = domain.isinside(Xinterior)
		Xinterior = Xinterior[I]

	if nboundary == 0:
		return Xinterior

	# Now we sample the boundary
	if len(domain) == 1:
		# Add the corners of the domain
		b1 = domain.corner(np.ones(1))
		b2 = domain.corner(-np.ones(1))
		Xbndry = np.vstack([b1, b2])
	elif Y.shape[1] == 1:
		b1 = domain.corner(L.flatten())
		b2 = domain.corner(-L.flatten())
		Xbndry = np.vstack([b1, b2])
	else:
		# Sample the boundary randomly
		# Generate directions in which we can sample
		if domain.A_eq.shape[0] > 0: 
			Q, _ = np.linalg.qr(domain.A_eq.T, mode = 'complete')
			# Dimension m x (m - # of constraints)
			Q = Q[:,domain.A_eq.shape[0]:]
		else:
			Q = np.eye(len(domain))

		if L is not None:
			# If an L is provided, we sample only the directions in the range of L
			I = np.argwhere(s> 1e-10).flatten()
			Q = Q.dot(Q.T.dot(VT[I,:].T))
			Q2, s2, _ = np.linalg.svd(Q, full_matrices = False)
			Q = Q2[:,np.argwhere(s2> 1e-10).flatten()]
		
		Z = sample_sphere(Q.shape[1], nboundary)
		QZ = np.dot(Q, Z.T).T
		center = domain.center
		Xbndry = np.array([center + qz*domain.extent(center, qz) for qz in QZ])

	return np.vstack([Xinterior, Xbndry])
	

if __name__ == '__main__':
	from domains import BoxDomain
	dom = BoxDomain([-1,-1,-1],[1,1,1])
	L = np.random.randn(len(dom)-2,len(dom))
	#L = np.eye(len(dom))
	X = dom.sample(10)
	Xhat = candidate_furthest_points(X, dom, L = L, nboundary = 0)
	print np.sort(cdist(np.dot(L,Xhat.T).T, np.dot(L,X.T).T), axis =1)
	print Xhat
