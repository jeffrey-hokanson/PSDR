from __future__ import print_function
import numpy as np
import scipy.linalg
import scipy.optimize
from scipy.spatial.distance import cdist, pdist, squareform
import cvxpy as cp


from .vertex import voronoi_vertex 
from .geometry import sample_sphere, unique_points, sample_simplex
from .domains import LinIneqDomain, ConvexHullDomain, SolverError

__all__ = ['seq_maximin_sample', 'fill_distance_estimate', 'initial_sample', 'Sampler', 'SequentialMaximinSampler',
	'multiobj_seq_maximin_sample', 'StretchedSampler']


def initial_sample(domain, L, Nsamp = int(1e4), Nboundary = 50):
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
		Samples that are well-distributed in L metric
	"""
	# Compute the active directions
	_, s, VT = scipy.linalg.svd(L)
	I = np.argwhere(~np.isclose(s,0)).flatten()
	U = VT.T[:,I]

	# An explicit, low-rank version of L
	Lhat = np.diag(s[I]).dot(U.T)

	Lrank = U.shape[1]

	if Lrank != len(domain):
		# Attempt to determine the effective dimension
		if Lrank == 1:
			# If L is rank 1, there is only one active subspace
			cs = np.array([domain.corner(U.flatten()), domain.corner(-U.flatten())])
			Lcs = Lhat.dot(cs.T).T
			dim = 1
		else:
			# Otherwise we first uniformly sample the rank-L dimensional sphere
			ds = sample_sphere(U.shape[1], Nboundary)
			# And then find points on the boundary in these directions
			# with respect to the active directions
			cs = np.array([domain.corner(U.dot(d)) for d in ds])
		
			# Construct a reduced-dimension L times the corners
			Lcs = Lhat.dot(cs.T).T
		
			# Remove duplicates
			I = unique_points(Lcs)
			cs = cs[I]
			Lcs = Lcs[I]
			
			# Compute the effective dimension using PCA on these points
			s2 = scipy.linalg.svdvals( (Lcs.T - np.mean(Lcs, axis = 0).reshape(-1,1) ))
			dim = np.sum(~np.isclose(s2,0))
		
			# Note that by computing the effective dimesion in this way
			# we remove the possibility that L might be rank-2
			# but that Lcs might be rank-1 due to equality constraints
			# in the active direction
	else:
		dim = len(domain)
		Lcs = []

	if dim == 1:
		assert len(Lcs) == 2
		
		# Even though these are on a line in the space, 
		# when vertex_sample with randomize=True, these points will be pushed off the line.
		alphas = np.random.uniform(0,1, size = Nsamp)
		X0 = np.vstack([alpha*cs[0] + (1-alpha)*cs[1] for alpha in alphas])	
		return X0
		
	elif dim in [2,3] and len(Lcs) > dim:
		# Construct a convex hull of low-dimensional points
		#if len(Lcs) > dim:
		#	# Insufficient points to describe hyperplane in m dimensions, instead return corners
		#	return cs
		Ldom = ConvexHullDomain(Lcs, solver = 'CVXOPT')
		# and sample uniformly here
		LX0 = Ldom.sample(Nsamp)
		
		# output points
		X0 = np.zeros((Nsamp, len(domain)))
		
		# We now find a (non-unique) point in the original domain
		# by solving a over-determined non-negative least squares problem.
		# Since it is over-determined we can add one row specifying the
		# convex-combination constraint and still recover the convex-combination
		# yielding Lx, and correspondingly push that back to the original domain 
		
		# Left hand side plus constraint
		A = np.vstack([ Lcs.T, np.ones( (1, len(Lcs) ))])
		for i in range(Nsamp):
			b = np.hstack([LX0[i], 1.])
			# find the convex combination
			alpha, ri = scipy.optimize.nnls(A, b)

			# check the solution
			#if np.linalg.norm(Lhat.dot(X0[i]) - LX0[i]) > 1e-4:
			if ri > 1e-4:
				print("Inexact solution")

			# project back onto the original space			
			X0[i] = cs.T.dot(alpha)
		
		return X0

	else:	
		# We are full rank and cannot do better than random sampling
		return domain.sample(Nsamp)
			

def seq_maximin_sample(domain, Xhat, L = None, Nsamp = int(1e3), X0 = None):
	r""" Performs one step of sequential maximin sampling. 

	Given an existing set of samples :math:`\lbrace \widehat{\mathbf{x}}_j\rbrace_{j=1}^M\subset \mathcal{D}`
	from the domain :math:`\mathcal{D} \subset \mathbb{R}^m`, this algorithm finds a point :math:`\mathbf{x} \in \mathcal{D}`
	that approximately solves the problem

	.. math::

		\max_{\mathbf{x} \in \mathcal{D}} \min_{j=1,\ldots,M} \|\mathbf{L}(\mathbf{x} - \widehat{\mathbf{x}}_j)\|_2.

	This algorithm uses :meth:`psdr.voronoi_vertex` to generate local maximizers of this problem
	and then returns the best of these.


	Parameters
	----------
	domain: Domain
		Domain on which to sample of dimension m
	Xhat: array-like (?, m)
		Existing samples on the domain
	L: array-like (?, m) optional
		Matrix defining the distance metric on the domain
	Nsamp: int, default 1e4
		Number of samples to use for vertex sampling
	X0: array-like (?, m)
		Samples from the domain to use in :meth:`psdr.voronoi_vertex`

	Returns
	-------
	x: np.ndarray(m)
		Sample from inside the domain
	"""
	Xhat = np.array(Xhat)
	
	if len(Xhat) == 0:
		# If we don't have any samples, pick one of the corners
		if L is None:
			return domain.corner(np.random.randn(len(domain)))
		else:
			_, s, VT = scipy.linalg.svd(L)
			return domain.corner(VT.T[:,0])
	
	Xhat = np.atleast_2d(Xhat)

	# Generate candidate points from the Voronoi vertices
	if X0 is None:
		if L is None:
			X0 = initial_sample(domain, np.eye(len(domain)), Nsamp = Nsamp)
		else:
			X0 = initial_sample(domain, L, Nsamp = Nsamp)

	Xcan = voronoi_vertex(domain, Xhat, X0, L = L, randomize = True)

	# Compute the Euclidean distance between candidates Xcan and current samples Xhat
	De = cdist(Xcan, Xhat)
	if L is not None:
		# If we have a non-trivial L matrix, also compute the distance here
		D = cdist(L.dot(Xcan.T).T, L.dot(Xhat.T).T)
	else:
		D = De

	# Find the distance to the closest neighbor
	d = np.min(D, axis = 1)
	de = np.min(De, axis = 1)

	# The index of the candidate point that is furthest
	# away from all its neighbors
	i = np.argmax(d)

	# Find points that are equivalent in distance 
	I = np.isclose(d[i], d)

	# Zero out their Euclidean distance we don't chose those	
	de[~I] = 0.

	# return the furthest point 
	i = np.argmax(de)

	return Xcan[i]


def multiobj_seq_maximin_sample(domain, Xhat, Ls, Nsamp = int(1e3)):
	r""" A multi-objective sequential maximin sampling 

	The goal of this algorithm is to return a new sample that maximizes
	the distance between samples in *several* different metrics.


	A typical use case will have Ls that are of size (1,m)
	
	"""

	Xhat = np.array(Xhat)
	if len(Xhat) == 0:
		Lall = np.vstack(Ls)
		return seq_maximin_sample(domain, Xhat, L = Lall, Nsamp = Nsamp) 

	vertices = []
	it = 0
	queue = []
	for k, L in enumerate(Ls):
		# Find initial samples well separated
		X0 = initial_sample(domain, L, Nsamp = Nsamp//len(Ls))
		
		# find the Voronoi vertices; we don't randomize as we are only interested
		# in the component that satisfies the constraint
		vertices = voronoi_vertex(domain, Xhat, X0, L = L, randomize = False) 
		
		# Remove duplicates in the L norm
		I = unique_points(L.dot(X0.T).T)
		vertices = vertices[I]

		# Compute the distances between points in this metric
		D = cdist(L.dot(vertices.T).T, L.dot(Xhat.T).T)
		D = np.min(D, axis = 1)
 
		for d, vertex in zip(D, vertices):
			# Place this point onto the priority queue
			# Note we include an iterator so that ties are always broken cleanly
			queue.append( (-d, it, k, vertex) )
			it += 1


	# Now greedily add constraints
	domain_samp = domain
	used = []

	queue.sort()
	for d, it, k, vertex in queue:
		if domain_samp.intrinsic_dimension == 0 or len(used) == len(Ls):
			break
	
		# We ignore constraints from L's we have already considered
		# since these must necessarily yield empty domains	
		if k not in used:
			L = Ls[k]
	
			# Try adding this equality constraint
			domain_test = domain_samp.add_constraints(A_eq = L, b_eq = L.dot(vertex))
			if not domain_test.empty:
				domain_samp = domain_test
				used.append(k)
#				print("appended %d" % k)
#				print(vertex)
#				print(L, "x =", L.dot(vertex))

	# Now sample the resulting domain 
	Lall = np.vstack(Ls)
	return seq_maximin_sample(domain_samp, Xhat, L = Lall, Nsamp = Nsamp)


def fill_distance_estimate(domain, Xhat, L = None, Nsamp = int(1e3), X0 = None ):
	r""" Estimate the fill distance of the points Xhat in the domain

	The *fill distance* (Def. 1.4 of [Wen04]_) or *dispersion* [LC05]_
	is the furthest distance between any point :math:`\mathbf{x} \in \mathcal{D}` 
	and a set of points 
	:math:`\lbrace \widehat{\mathbf{x}}_j \rbrace_{j=1}^m \subset \mathcal{D}`:
		
	.. math::

		\sup_{\mathbf{x} \in \mathcal{D}} \min_{j=1,\ldots,M} \|\mathbf{L}(\mathbf{x} - \widehat{\mathbf{x}}_j)\|_2.

	Similar to :meth:`psdr.seq_maximin_sample`, this uses :meth:`psdr.voronoi_vertex` to find 
	a subset of local maximizers and returns the best of these.

	Parameters
	----------
	domain: Domain
		Domain on which to compute the dispersion
	Xhat: array-like (?, m)
		Existing samples on the domain
	L: array-like (?, m) optional
		Matrix defining the distance metric on the domain
	Nsamp: int, default 1e4
		Number of samples to use for vertex sampling
	X0: array-like (?, m)
		Samples from the domain to use in :meth:`psdr.voronoi_vertex`

	Returns
	-------
	d: float
		Fill distance lower bound

	References
	----------
	..  [Wen04] Scattered Data Approximation. Holger Wendland.
			Cambridge University Press, 2004.
			https://doi.org/10.1017/CBO9780511617539
	.. [LC05] Iteratively Locating Voronoi Vertices for Dispersion Estimation
		Stephen R. Lindemann and Peng Cheng
		Proceedings of the 2005 Interational Conference on Robotics and Automation
	"""
	
	if X0 is None and L is None:
		X0 = domain.sample(Nsamp)
	elif X0 is None and L is not None:
		X0 = initial_sample(domain, L, Nsamp = Nsamp)

	# Since we only care about distance in L, we can terminate early if L is rank-deficient
	# and hence we turn off the randomization
	Xcan = voronoi_vertex(domain, Xhat, X0, L = L, randomize = False)

	# Euclidean distance
	if L is not None:
		D = cdist(L.dot(Xcan.T).T, L.dot(Xhat.T).T)
	else:
		D = cdist(Xcan, Xhat)

	d = np.min(D, axis = 1)
	return float(np.max(d))

	
class Sampler:
	r""" Generic sampler interface

	Parameters
	----------
	fun: Function
		Function for which to preform a design of experiments
	X: array-like (?,m)
		Existing samples from the domain
	fX: array-like (?,nfun)
		Existing evaluations of the function at the points in X
	"""
	def __init__(self, fun, X = None, fX = None):
		self._fun = fun
		
		if X is None:
			X = np.zeros((0, len(fun.domain)))
		else:
			X = np.copy(np.array(X))
			assert X.shape[1] == len(fun.domain), "Input dimensions do not match function"
		
		self._X = X
		
		if fX is not None:
			fX = np.copy(np.array(fX))
			assert fX.shape[0] == X.shape[0], "Number of function values does not match number of samples"

		self._fX = fX

	def sample(self, draw = 1, verbose = False):
		r""" Sample the function


		Parameters
		----------
		draw: int, default 1
			Number of samples to take
		"""
		return self._sample(draw = draw, verbose = verbose)
	
	def sample_async(self, draw = 1):
		r""" Sample the function asynchronously updating the search parameters


		Parameters
		----------
		draw: int, default 1
			Number of samples to take
		"""
		return self._sample_async(draw = draw)

	def _sample(self, draw = 1, verbose = False):
		raise NotImplementedError
	
	def _sample_async(self, draw = 1):
		raise NotImplementedError

	@property
	def X(self):
		r""" Samples from the function's domain"""
		return self._X.copy()

	@property
	def fX(self):
		r""" Outputs from the function corresponding to samples X"""
		if self._fX is not None:
			return self._fX.copy()
		else:
			return None
	
class SequentialMaximinSampler(Sampler):
	r""" Sequential maximin sampling with a fixed metric

	Given a distance metric provided by :math:`\mathbf{L}`,
	construct a sequence of samples :math:`\widehat{\mathbf{x}}_i`
	that are local solutions to


	.. math::
		
		\widehat{\mathbf{x}}_j = \arg\max_{\mathbf{x} \in \mathcal{D}} \min_{i=1,\ldots,j}
			 \|\mathbf{L}(\mathbf{x} - \widehat{\mathbf{x}}_i)\|_2.

	
	Parameters
	----------
	fun: Function
		Function for which to preform a design of experiments
	L: array-like (?, m)
		Matrix defining the metric
	X: array-like (?,m)
		Existing samples from the domain
	fX: array-like (?,nfun)
		Existing evaluations of the function at the points in X

	"""
	def __init__(self, fun, L = None, X = None, fX = None):
		Sampler.__init__(self, fun, X = X, fX = fX)
		if L is None:
			L = np.eye(len(fun.domain))
		else:
			L = np.atleast_2d(np.array(L))
			assert L.shape[1] == len(fun.domain), "Dimension of L does not match domain"
		self._L = L

	def _sample(self, draw = 1, verbose = False):
		Xnew = []
		# As L is fixed, we can draw these samples at once
		for i in range(draw):
			xnew = seq_maximin_sample(self._fun.domain, self._X, L = self._L)
			Xnew.append(xnew)
			if verbose:
				print('%3d: ' % (i,),  ' '.join(['%8.3f' % x for x in xnew]))
			self._X = np.vstack([self._X, xnew])

		# Now we evaluate the function at these new points
		# (this takes advantage of potential vectorization of fun
		fXnew = self._fun.eval(Xnew)
		if self._fX is None:
			self._fX = fXnew
		else:
			if len(fXnew.shape) > 1:
				self._fX = np.vstack([self._fX, fXnew])
			else:
				self._fX = np.hstack([self._fX, fXnew])


class StretchedSampler(Sampler):
	r"""

	"""
	def __init__(self, fun, X = None, fX = None, pras = None, funmap = None):
		Sampler.__init__(self, fun, X = X, fX = fX)
		self._pras = pras 

		if funmap is None:
			funmap = lambda x: x
		self._funmap = funmap


	def _sample(self, draw = 1):
		for it in range(draw):
			return self._sample_one()

	def _sample_one(self):
		 pass


#class Sampler(object):
#	def __init__(self, f, domain, pool = None, X0 = None, fX0 = None):
#		
#		# Copy over variables
#		self.f = f
#		self.domain = domain
#		
#		if pool is None:
#			pool = SequentialPool()
#		self.pool = pool
#
#		if X0 is None:
#			self.X = []
#		else:
#			self.X = [x for x in X0]
#		if fX0 is None:
#			self.fX = []
#		else:
#			self.fX = [fx for fx in fX0]
#		
#
#	def _draw_sample(self, Xrunning):
#		raise NotImplementedError
#
#	def sample(self, draw = 1):
#		""" 
#		"""
#		for k in range(draw):
#			Xrunning = np.zeros((0, len(self.domain))) 
#			xnew = self._draw_sample([Xrunning,])
#			job = self.pool.apply(self.f, args = [xnew,])
#			fxnew = job.output
#			self.X  += [xnew]
#			self.fX += [float(fxnew)]
#			
#
#	def parallel_sample(self, draw = 1, dt = 0.1):
#		# TODO: Add assertion about pool support async 
#		njobs = 0
#		jobs = []
#	
#		while njobs < draw:
#			# If we have a worker avalible 
#			if self.pool.avail_workers() > 0:
#				# Determine which jobs are done
#				done = [k for k, job in enumerate(jobs) if job.ready()]
#
#				# Get the updated information
#				Xnew = [jobs[k].args[0] for k in done]
#				fXnew = [jobs[k].output for k in done]
#
#				# Update the input/ouput pairs
#				self.X  += Xnew
#				self.fX += fXnew
#
#				# Delete the completed jobs
#				jobs = [jobs[k] for k in enumerate(jobs) if k not in done]
#				
#				# Extract the points that are currently running
#				Xrunning = [jobs.args[0] for k in jobs] 
#				
#				# Draw a sample and enqueue it
#				x = self._draw_sample(Xrunning)
#				jobs.append(self.pool.apply_async(self.f, args = [x,]))
#				njobs += 1	
#			else:
#				time.sleep(dt)
#
#		# Wait for all jobs to finish
#		self.pool.join()		
#				
#		Xnew  = [job.args[0] for job in jobs]
#		fXnew = [job.output for job in jobs]
#
#		# Update X, fX
#		self.X += Xnew
#		self.fX += fXnew
#
#
#
#class RandomSampler(Sampler):
#	def _draw_sample(self, Xrunning):
#		return self.domain.sample()	
#
#class UniformSampler(Sampler):
#	""" Sample uniformly on the normalized domain
#
#	"""
#	def __init__(self, f, domain, pool = None, X0 = None, fX0 = None):
#		Sampler.__init__(self, f, domain, pool = pool, X0 = X0, fX0 = fX0) 		
#		self.domain_norm = self.domain.normalized_domain()
#		self.L = np.eye(len(self.domain))
#
#	def _draw_sample(self, Xrunning):
#		Xall = np.array(self.X + Xrunning)
#		Xall_norm = self.domain.normalize(Xall)
#		
#		xnew_norm = maximin_sample(Xall_norm, self.domain_norm, self.L)
#		xnew = self.domain.unnormalize(xnew_norm)
#		return xnew
#
#class RidgeSampler(Sampler):
#	"""
#
#	Note: the ridge approximation is always formed on the normalized domain
#
#	Parameters
#	----------
#	pra: Instance of PolynomialRidgeApproximation
#	"""
#	def __init__(self, f, domain, pra, **kwargs):
#		Sampler.__init__(self, f, domain, **kwargs)
#		self.domain_norm = self.domain.normalized_domain()
#		self.pra = pra
#		self.U = None
#		self.fill_dist = np.inf
#
#	def _draw_sample(self, Xrunning):
#		if len(self.fX) <= len(self.domain)+1:
#			return self.domain.sample()
#
#		# Build ridge approximation
#		X_norm = self.domain.normalize(self.X)
#		fX = np.array(self.fX)
#		I = np.isfinite(fX)
#		try:
#			self.pra.fit(X_norm[I], fX[I])
#		except (UnderdeterminedException, IllposedException):
#			# If we can't yet solve the problem, sample randomly
#			return self.domain.sample()		
#
#		Xall = np.vstack(self.X + Xrunning)
#		Xall_norm = self.domain.normalize(Xall)
#		self.U = self.pra.U
#		xnew_norm = maximin_sample(Xall_norm, self.domain_norm, L = self.U.T)
#		self.fill_dist = np.min(cdist(self.U.T.dot(xnew_norm).reshape(1,-1), self.U.T.dot(Xall_norm.T).T))
#		xnew = self.domain.unnormalize(xnew_norm)
#		return xnew
#					
#if __name__ == '__main__':
#	from demos import golinski_volume, build_golinski_design_domain
#	from poly_ridge import PolynomialRidgeApproximation
#	dom = build_golinski_design_domain()
#		
#	f = golinski_volume
#	
#	np.random.seed(0)
#
#	pra = PolynomialRidgeApproximation(degree = 5, subspace_dimension = 1)
#	samp = RidgeSampler(f, dom, pra)
#	samp.sample(2)
#	for k in range(2,1000):
#		samp.sample()
#		print("%3d %5.2e" % (k, samp.fill_dist))
		 
