from __future__ import division, print_function

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
	if domain.is_point:
		return domain.sample(1)	

	# An explicit, low-rank version of L
	J = low_rank_L(L)

	Lrank = J.shape[0]

	# Make sure we sample enough points on the boundary to preserve the full dimension
	Nboundary = max(Nboundary, Lrank + 1)

	if Lrank == 1:
		# If L is rank 1 then the projection of the domain is an interval
		cs = np.array([domain.corner(J.flatten()), domain.corner(-J.flatten())])
		Jcs = J.dot(cs.T).T
	else:
		# Otherwise we first uniformly sample the rank-L dimensional sphere
		zs = sample_sphere(Lrank, Nboundary)
		# And then find points on the boundary in these directions
		# with respect to the active directions
		U, _ = np.linalg.qr(J.T)
		cs = np.array([domain.corner(U.dot(z)) for z in zs])
		# Construct a reduced-dimension L times the corners
		Jcs = J.dot(cs.T).T
	
		# Remove duplicates
		I = unique_points(Jcs)
		cs = cs[I]
		Jcs = Jcs[I]
		
		# Compute the effective dimension using PCA on these points
		
		# As we will later map back to the original domain
		# through the convex combination of these points,
		# we center these points to improve conditioning
		Jcs = (Jcs.T - np.mean(Jcs, axis = 0).reshape(-1,1)).T

		# We now do PCA on these centered points (i.e., SVD)
		# to assess if these have been projected onto a low dimensional manifold
		U2, s2, VT2 = scipy.linalg.svd(Jcs, full_matrices = False)
		dim = np.sum(~np.isclose(s2,0))
		# If they are on this manifold, rotate onto this basis
		if dim < Lrank:
			V2 = VT2.T
			Jcs = Jcs.dot(V2[:,0:dim])
	
	
	if len(Jcs) == 2:
		# If there only two points on the boundary, 
		# we have an interval that we can easily sample
		alphas = np.random.uniform(0,1, size = Nsamp)
		X0 = np.vstack([alpha*cs[0] + (1-alpha)*cs[1] for alpha in alphas])	

	else:
		# Otherwise we sample the convex hull of the projected corners
		Jdom = ConvexHullDomain(Jcs)
			
		# Sampling a linear inequailty domain is much faster than a convex hull domain
		# so if this is feasible, we convert to this format
		if len(Jdom) <= 3:
			Jdom_ineq = Jdom.to_linineq()
			ws = Jdom_ineq.sample(Nsamp)
		else:
			ws = Jdom.sample(Nsamp)

		# And project those points back into the original domain
		X0 = np.zeros( (Nsamp, len(domain)) )
		for i, w in enumerate(ws):
			# Determine the combination coefficients for these samples
			alpha = Jdom.coefficients(w)
			X0[i] = cs.T.dot(alpha)

	return X0
