""" Functions for constructing efficient samples of ridge functions


"""
import numpy as np
from opt import linprog, LinProgException 
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from scipy.linalg import orth
from domains import ConvexHullDomain, LinIneqDomain, BoxDomain, EmptyDomain

import sys

from parallel import pmap
from tqdm import tqdm


def constrain_domain(domain, A = None, b = None, A_eq = None, b_eq = None):
	""" Return a domain with the specified constraints added
	"""
	m = len(domain)
	if A is None:
		A = np.zeros((0,m))
	if b is None:
		b = np.zeros((0,))
	if A_eq is None:
		A_eq = np.zeros((0,m))
	if b_eq is None:
		b_eq = np.zeros((0,))

	A = np.vstack([domain.A, A])
	b = np.hstack([domain.b, b])
	A_eq = np.vstack([domain.A_eq, A_eq])
	b_eq = np.hstack([domain.b_eq, b_eq])
	return LinIneqDomain(A = A, b = b, lb = domain.lb, ub = domain.ub, A_eq = A_eq, b_eq = b_eq) 

def sample_sphere(dim, n, k = 100):
	""" Sample points on a high-dimensional sphere 

	Uses Mitchell's best candidate algorithm.
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

def sample_uniform(domain, X0 = None, M = None, target_separation = None, mitchell_sample = 100, verbose = False):
	""" Sample the domain approximately evenly using Mitchell's best candidate algorithm
	
	Parameters
	----------
	domain: Domain
		Domain from which to sample
	X0: np.ndarray (M,m)
		Previous samples which to include in the distance comparison step
	M: int [optional]
		Number of samples to generate
	target_separation: float [optional]
		Stop generating samples when the distance between these samples divided by domain radius
		drops below this value
	mitchell_sample: int [optional]
		Number of samples to take in Mitchell's best candidate algorithm
	verbose: bool [optional]
		Display messages
			
	"""
	assert M is not None or target_separation is not None, "One of n or target_separation must be specified"
	mitchell_sample = int(mitchell_sample)

	if target_separation is None:
		target_separation = 0.
	
	if M is None:
		M = np.inf	
	else:
		M = int(M)

	m = len(domain)
	
	if M == 0:
		return np.zeros((0,m))
	
	if X0 is None:
		X0 = np.zeros((0, m))
	else:
		assert len(X0.shape) ==2, "X0 should be of shape (number samples, dimension of domain)"
		assert X0.shape[1] == m, "Dimension of X0 should match that of the domain"

	X = np.zeros((0,m))
	while True:
		# Points to compare against
		Xcomp = np.vstack([X0, X])
		
		if Xcomp.shape[0] == 0:
			# If we have nothing to compare against, sample one point randomly
			Xcan = domain.sample(1)
			X = Xcan.reshape(1,-1)
			dist = np.inf	

		else:
			Xcan = domain.sample(mitchell_sample)
			if mitchell_sample == 1:
				Xcan = Xcan.reshape(1,-1)
		
			dist = np.min(cdist(Xcomp, Xcan), axis = 0)
			I = np.argmax(dist)
			dist = dist[I]
			if target_separation > dist:
				break
			X = np.vstack([X, Xcan[I].reshape(1,-1)])
			
		if verbose:
			print "%4d samples; max min distance %1.2e" % (X.shape[0], target_separation/domain.radius)
		if X.shape[0] >= M or target_separation > dist:
			break	
	return X




def sample_ridges(domain, Us, M = None, target_separation = None, verbose = True, X0 = None, mitchell_sample = 100, M_interior = 10):
	""" Sample a domain along multiple ridges simultaneously
	
	Parameters
	----------
	domain: Domain
		Domain from which to sample of dimension m

	Us: list of np.ndarrays of size (m,n) 
		Ridge directions
	
	M: int
		Maximum number of samples to construct

	target_separation: float [optional]
		Target separation of points in each ridge direction, 
		relative to each projected domains radius

	X0: np.ndarray of size (M,m) [optional]
		Previously selected points

	verbose: bool [optional]
		If true, show messages
	
	M_interior: int [optional]
		If target_separation is not provided, take this many samples per ridge direction
		
	"""
	m = len(domain)
	
	if X0 is None:
		X0 = np.zeros((0, m))
	else:
		assert len(X0.shape) ==2, "X0 should be of shape (number samples, dimension of domain)"
		assert X0.shape[1] == m, "Dimension of X0 should match that of the domain"

	# If ridge directions have been supplied as a vector, reshape to matrices
	for i in range(len(Us)):
		if len(Us[i].shape) == 1:
			Us[i] = Us[i].reshape(-1,1)

	if verbose:
		print "Building ridge domains"
	
	ridge_domains = [build_ridge_domain(domain, U) for U in Us]

	if M is None:
		M = np.inf	

	# Build space to hold samples
	X = np.zeros((0,m))

	while True:
		# Points to compare against
		Xcomp = np.vstack([X0, X])
		
		# Which ridge directions are being used
		active_ridges = [False]*len(Us)

		# Pick points where we want to sample
		Uy_list= []
		dists = []
		for ridge_domain, U in zip(ridge_domains, Us):
			Y0 = np.dot(U.T, Xcomp.T).T
			# We perform a rather exhaustive sample of these low dimensional domains since this is cheap
			y = sample_uniform(ridge_domain, M = 1, X0 = Y0, mitchell_sample = 10*10**len(ridge_domain) )
			if Y0.shape[0] > 0:
				dist = cdist(y, Y0)
				dist = np.min(dist)
			else:
				dist = np.inf
			Uy_list.append( (U, y[0]) )
			dists.append(dist/ridge_domain.radius)
		
		# Determine which ridge needs to be sampled the most
		I = np.argmax(dists)
		#print "Selected ridge %d" % I	
		U0, y0 = Uy_list[I]
		max_dist = dists[I]
		active_ridges[I] = True
		#print "y0", y0	
		Uy_constraints = [(U0, y0)]
		
		if target_separation > max_dist:
			if verbose:
				print "Stopping because target density achieved"
			break

		# Now select points which to sample
		Uy_list = []
		for ridge_domain, U in zip(ridge_domains, Us):
			if np.linalg.norm(U - U0, 'fro') > 1e-10:
				# Only construct points that are not in our first constraint
				Y0 = np.dot(U.T, Xcomp.T).T
				if target_separation is not None:
					ys = sample_uniform(ridge_domain, target_separation = target_separation, mitchell_sample = mitchell_sample, X0 = Y0)
				else:
					ys = sample_uniform(ridge_domain, M = M_interior, mitchell_sample = mitchell_sample, X0 = Y0)
				
				for y in ys:
					Uy_list.append((U, y))

		n_candidates = len(Uy_list)
		#print "Found %d candidates" % (n_candidates,)	
		#while len(Uy_list) > 0:
		while False:
			# Do a greedy search to determine which constraint to add
			radii = np.zeros(len(Uy_list))
			for k, (U, y) in tqdm(enumerate(Uy_list), total = len(Uy_list), desc = "Combinations"):
				A_eq = np.vstack([np.atleast_2d(U0.T) for U0,y0 in Uy_constraints] + [np.atleast_2d(U.T)])
				b_eq = np.hstack([y0 for U0,y0 in Uy_constraints] + [y])
				radii[k] = domain_radius(domain, A_eq, b_eq)
			
			print "radii", radii
	
			I = np.argmax(radii)
			if radii[I] > 0:
				U0, y0 = Uy_list.pop(I)
				Uy_constraints.append( (U0, y0) )
				# Remove U that are the same as the constraint we just added
				Uy_list = [ (U,y) for (U,y) in Uy_list if np.linalg.norm(U - U0, 'fro') > 1e-10]
				k = int(np.argwhere([np.linalg.norm(U - U0, 'fro') < 1e-10 for U in Us])[0])
				active_ridges[k] = True
				#print "Combined %d constraints" % len(Uy_constraints)
			else:
				break

		if verbose:
			print "sample #%4d; using %2d constraints from %3d candidates; max dist %1.2e" % (
				X.shape[0], len(Uy_constraints), n_candidates, max_dist), "active ridges", np.argwhere(active_ridges).flatten()
		
		A_eq = np.vstack([np.atleast_2d(U.T) for U,y in Uy_constraints])
		b_eq = np.hstack([np.atleast_1d(y) for U,y in Uy_constraints])
	
		# Sample this point
		domain_constrained = constrain_domain(domain, A_eq = A_eq, b_eq = b_eq)
		X_new = sample_uniform(domain_constrained, M = 1, mitchell_sample = mitchell_sample, X0 = Xcomp)
		X = np.vstack([X, X_new])
		if X.shape[0] >= M:
			break
	return X



def extent(dom, U, z):
	c = np.dot(U, z)
	x = linprog(c, A_ub = dom.A, b_ub = dom.b, lb = dom.lb, ub = dom.ub, A_eq = dom.A_eq, b_eq = dom.b_eq)
	return np.dot(U.T,x)

def domain_radius(domain, A_eq, b_eq):
	try:
		inner_dom = constrain_domain(domain, A_eq = A_eq, b_eq = b_eq)
		return inner_dom.radius
	except:
		return 0.


def random_voronoi_vertex(domain, X, L = None):
	""" Compute a random vertex of the Voronoi diagram associated with the set of points

	Computes one vertex (intersection of multiple Voronoi cells) randomly


	TODO: Replace with randomized algorithm with reduced complexity cost,
	as per discussion with Drew Kouri

	Parameters
	----------
	domain: Domain
		Specifies the space in which X lives
	X: np.ndarray(M,n)
		Current samples from the domain
	L: np.ndarray(n,n)
		Metric associated the desired Voronoi diagram 
	""" 
	if L is None:
		L = np.eye(X.shape[1])

	i = np.random.randint(X.shape[0])
	xi = X[i]
	# Compute the linear inequalities associated with this point
	A = []
	b = []
	for j, xj in enumerate(X):
		if i != j:
			b.append( -0.5*np.dot( np.dot(L, xi - xj).T, np.dot(L, xi + xj)))
			A.append( -np.dot( np.dot(L, xi - xj).T, L)) 
	A = np.vstack(A)
	b = np.hstack(b)

	# Build up the domain associated with the Voronoi cell
	cell = domain.add_constraint(A = A, b = b, center = xi)
	r = np.random.randn(len(cell))
	return cell.corner(r)



# Keep this
def voronoi_vertices(Y):
	""" Compute all the Voronoi vertices 

	This function provides a uniform access to the Voronoi vertices,
	those points in n-dimensional space that are equidistant from n+1 points.
	
	This is necessary since QHull, and consequently scipy.spartial.Voronoi
	only can handle 2-dimensional and higher 

	Parameters
	----------
	Y: np.ndarray(M, n)
		Array of points 
	
	Returns
	-------
	np.ndarray(N, n)
		points far away from X in the projected domain
	"""
	if len(Y.shape) ==1 or Y.shape[1] == 1:
		# Q-hull doesn't handle the 1d case because it is straightfoward
		yhat = np.sort(Y.flatten())
		return 0.5*(yhat[1:] + yhat[:-1]).reshape(-1,1)
	else:
		vor = Voronoi(Y)
		return vor.vertices 

# Keep this
def zonotope(domain, U, n_sample = None):
	""" Construct an approximate zonotope 
	"""

	if n_sample is None:
		n_sample = 10*4**U.shape[1]

	# Construct samples of boundary
	if U.shape[1] == 1:
		# With a one dimensional ridge, we only need to consider +/- U
		X = np.vstack([ domain.corner(U.flatten()), domain.corner(-U.flatten())])
	else:
		Z = sample_sphere(U.shape[1], n_sample)
		X = np.vstack([ domain.corner(np.dot(U, z)) for z in Z])

	# Compress down onto the ridge
	Y = np.dot(U.T, X.T).T
	return ConvexHullDomain(Y)


# Keep this
def stretch_sample(domain, Us, X0 = None, M = 1, verbose = False, enrich = True, M_interior = 100):
	""" Sample greedly from a set of ridge directions

	Stretch sampling chooses a new sample 

	Parameters
	----------
	domain: Domain
		Space on which to sample
	Us: list of np.arrays 
		List of subspaces on which to sample
	X0: [optional] np.array(M, m)
		List of samples already taken from the space
	M: [optional] int, default: 1
		Number of samples to take
	verbose: [optional] bool, default: False
		If True, explain choices at each step
	"""

	if X0 is None:
		X0 = np.zeros((0, len(domain)))

	X1 = np.zeros((0, len(domain)))

	# Construct the zonotope for each domain
	zts = [zonotope(domain, U) for U in Us]

	for i in range(M):
		# For each direction, find the point furthest away from other points
		Xall = np.vstack([X0, X1])
		
		Ybest = []
		maxdist = []
		boundary_can = []
		message = ''
		for zt, U in zip(zts, Us):
			# Project current points onto this subspace
			Yall = np.dot(U.T, Xall.T).T
			
			# Construc the candidates 
			Yboundary = zt.X # samples from the boundary we used to make the zonotope
			Yvoronoi = voronoi_vertices(np.vstack([Yall, zt.vertices]))
			Ycan = np.vstack([Yboundary, Yvoronoi])

			# Pick the point that has the maximum minimum distance
			dist = np.min(cdist(Ycan, Yall, 'euclidean'), axis = 1)
			j = np.argmax(dist)

			# Record if sampled from voronoi or zonotope 
			boundary_can.append(j < len(Yboundary))
			Ybest.append(Ycan[j])
			maxdist.append(dist[j])

		# Choose the ridge direction with furthest away point
		j = np.argmax(maxdist)
		maxdist = maxdist[j]
		U_best = Us[j]
		y_best = Ybest[j]
		boundary_can = boundary_can[j]
	
		message = 'Step %3d using U %3d, dist %5.5f, ' % (i, j, maxdist)

		if boundary_can:
			# If we are a candidate from the boundary,
			# find the corner in that direction
			x_new1 = domain.corner(np.dot(U_best, y_best))		
			x_new2 = domain.corner(-np.dot(U_best, y_best))	
			if np.linalg.norm(np.dot(U_best.T, x_new1) - y_best) < np.linalg.norm(np.dot(U_best.T, x_new2) - y_best):
				x_new = x_new1
			else:
				x_new = x_new2
			message += 'sampled boundary '
		else:
			# Otherwise, we add this equality constraint to the domain
			# and pick a far away sample
			try:
				# Try constructing the domain by adding an equality constraint
				sample_dom = domain.add_constraint(A_eq = U_best.T, b_eq = y_best)
			except EmptyDomain:
				# If we can't form this domain (likely due to numerical artifacts near the boundary)
				# find the closest point inside the domain by solving a quadratic program
				x_new = domain.projected_closest_point(U_best.T, y_best).reshape(-1)
				message += 'sampled closest  '
			else:	
				message += 'sampled interior '
			
				if enrich:
					# Generate candidate directions on each of the ridges
					ps = []
					for U in [Us[k] for k in range(len(Us)) if k != j]:
						if U.shape[1] == 1:
							ps.append(U.flatten())
							ps.append(-U.flatten())
						else: 	
							ps += [np.dot(U, z) for z in sample_sphere(U.shape[1], 10)]

					# Generate list of directions of the desired length
					I = np.random.permutation(len(ps))
					ps = [ps[k] for k in I[0:M_interior - 1]]
					Xcan = [sample_dom.corner(p) for p in ps] + [sample_dom.center.flatten()]
					if len(Xcan) < M_interior:
						Xcan += [x for x in sample_dom.sample(M_interior - len(Xcan))]
					Xcan = np.vstack(Xcan)
				else:
					Xcan = sample_dom.sample(M_interior)

				# Pick the point satisfying this constraint furthest away
				dist = np.min(cdist(Xcan, Xall), axis = 1)
				k = np.argmax(dist)
				x_new = Xcan[k].reshape(-1)

		if not domain.isinside(x_new):
			if verbose:
				print "point not inside" 
			x_new = domain.closest_point(x_new).reshape(-1)
		
		# Compute distance
		if verbose:
			measdist = np.min(cdist(np.dot(U_best.T, x_new).reshape(1,-1) ,np.dot(U_best.T, Xall.T).T))
			message += 'measured dist %5.5e' % (measdist,) 
	
		X1 = np.vstack([X1, x_new.reshape(1,-1)])

		if verbose:
			print message

	return X1
	

if __name__ == '__main__':
	m = 5
	A = np.zeros((1,m))
	A[0,0] = 1
	A[0,1] = 1
	b = -1*np.ones((1,))
	dom = LinIneqDomain(A, b, lb = -1*np.ones(m), ub = 1*np.ones(m))
	X = dom.sample(1000)
	
	#L = np.zeros((2,m))
	#L[0,0] = 1
	#L[1,1] = 1
	#x = random_voronoi_vertex(dom, X, L)
	#print x

	U = orth(np.random.randn(m,2))
	Xcan = voronoi_vertices(np.dot(U.T, X.T).T)
	#print Xcan.shape
	#dom2 = zonotope(dom, U)
	#print dom2.vertices
	stretch_sample(dom, [U], X, M = 10, verbose = True)


if False:# __name__ == '__main__':
	from domains import *
	import matplotlib.pyplot as plt
	#demo_sample()
	
	m = 5
	A = np.zeros((1,m))
	A[0,0] = 1
	A[0,1] = 1
	b = -1*np.ones((1,))
	dom = LinIneqDomain(A, b, lb = -1*np.ones(m), ub = 1*np.ones(m))


	#X0 = sample_uniform(dom, M = 10, verbose = True, mitchell_sample = 10)
	#X = sample_uniform(dom, M = 10, X0 = X0, verbose = True, mitchell_sample = 10)
	
	U0 = np.zeros(m)
	U0[0] = 1.
	U0[1] = -1.
	U0 /= np.linalg.norm(U0)
	U1 = np.zeros(m)
	U1[1] = 1
	U1[0] = 1.
	U1 /= np.linalg.norm(U1)
	Us = [U0, U1]
	#X = sample_ridges(dom, Us, target_separation = None, M = 100, M_interior = 0)
	
	fig, ax = plt.subplots()
	ax.hist(np.dot(U0.T, X.T).T)
	rdom = build_ridge_domain(dom, U0)
	rng = rdom.range(np.ones(1))
	ax.axvline(rng[0], color = 'k')
	ax.axvline(rng[1], color = 'k')
	fig, ax = plt.subplots()
	ax.hist(np.dot(U1.T, X.T).T)
	rdom = build_ridge_domain(dom, U1)
	rng = rdom.range(np.ones(1))
	ax.axvline(rng[0], color = 'k')
	ax.axvline(rng[1], color = 'k')
	plt.show()
