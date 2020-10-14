import numpy as np
import scipy

from .util import low_rank_L
from ..geometry import sample_sphere, unique_points
from ..domains import ConvexHullDomain

def initial_sample(domain, L, Nsamp = int(1e2), Nboundary = 50):
	r""" Construct initial points for a low-rank L matrix

	The Voronoi vertex sampling algorithm :meth:`psdr.voronoi_vertex`
	requires an initial set of points which are then pushed towards Voronoi vertices.
	In high dimensional spaces with a low-rank distance metric given by L, random sampling
	is going to concentrate samples in this metric and ignore the boundaries.
	This will make the Voronoi vertices sampled by this algorithm tend to ignore
	areas near the boundary and hence, these will provide poor samples for a maximin
	design of experiments. This algorithm attempts to sample a high-dimensional space
	with a low-rank pseudo-metric  

	Parameters
	----------
	domain: Domain
		Domain on which to sample
	L: array-like (?,m)
		Matrix defining the (semi)-metric for the space
	Nsamp: int, optional
		Number of samples to return
	Nboundary: int, optional
		Number of samples to take from the boundary for constructing the boundary
		of the domain projected onto a low-rank L

	Returns
	-------
	X0: np.ndarray(Nsamp, m)
		Samples that are well-distributed in L metric in domain
	"""
	# If we are sampling with a scalar multiple of the identity matrix,
	# we can do no better than random sampling on the original domain
	if L.shape[0] == L.shape[1]:
		err = np.linalg.norm(L[0,0]*np.eye(L.shape[0]) - L)
		if np.isclose(err, 0):
			return domain.sample(Nsamp)

	# If there is only one point in the domain,
	# we simply return that one point
	# TODO: This should be combined with a determination if the domain is a point
	# since if not initialized, this will call corner 2m times.

	# An explicit, low-rank version of L
	L = low_rank_L(L)

	Lrank = L.shape[0]

	# Make sure we sample enough points on the boundary to preserve the full dimension
	Nboundary = max(Nboundary, Lrank + 1)

	if Lrank == 1:
		zs = [-np.ones(1), np.ones(1)]
	else:
		zs = sample_sphere(Lrank, Nboundary)

	U, _ = np.linalg.qr(L.T)
	# Find corners with respect to directions in L
	cs = np.array([domain.corner(U @ z) for z in zs])
	# Multiply by the low-rank Lipschitz matrix 
	Lcs = (L @ cs.T).T
	# Remove duplicates (although done in ConvexHullDomain, we need these unique points to reconstruct the points) 
	I = unique_points(Lcs)
	cs = cs[I]
	Lcs = Lcs[I]


	# Create a domain using these points
	Ldom = ConvexHullDomain(Lcs)
	# Determine which points were used
	Y = Ldom.sample(Nsamp)

	# Convert back to the ambient space
	X = np.zeros((Nsamp, len(domain)))
	for i, y in enumerate(Y):
		# Determine the combination coefficients for these samples
		alpha = Ldom.coefficients(y)
		X[i] = cs.T @ alpha

	return X	


