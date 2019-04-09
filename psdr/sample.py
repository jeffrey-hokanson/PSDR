from __future__ import print_function
import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist, pdist, squareform

from .vertex import voronoi_vertex 
from .geometry import sample_sphere, unique_points, sample_simplex
from .domains import LinIneqDomain

__all__ = ['seq_maximin_sample', 'fill_distance_estimate', 'initial_sample']


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
		Number of samples to take from the boundary

	Returns
	-------
	X0: np.ndarray(Nsamp, m)
		Samples that are well-distributed in L metric
	"""
	# Compute the active directions
	_, s, VT = scipy.linalg.svd(L)
	I = np.argwhere(~np.isclose(s,0)).flatten()
	U = VT.T[:,I]
	Lhat = np.diag(s[I]).dot(U.T)
	
	if U.shape[1] == len(domain):
		# We are full rank and cannot do better than random sampling
		return domain.sample(Nsamp)
	
	elif U.shape[1] == 1:
		# If we have a one dimensional space, we only need to find the corners
		# and sample uniformly between them
		c1 = domain.corner(U.flatten())
		c2 = domain.corner(-U.flatten())
		
		# Even though these are on a line in the space, 
		# when vertex_sample with randomize=True, these points will be pushed off the line.
		#alphas = np.linspace(0,1, Nsamp)
		#alphas[1:-1] += np.random.uniform(-1./Nsamp, 1/Nsamp, size = (Nsamp - 2))
		alphas = np.random.uniform(0,1, size = Nsamp)
		X0 = np.vstack([alpha*c1 + (1-alpha)*c2 for alpha in alphas])	
		return X0

	else:
		# In this case we sample uniformly from the interior 
		# First we uniformly sample the rank-L dimensional sphere
		ds = sample_sphere(U.shape[1], Nboundary)
		# These are points on the corners of the domain
		cs = [domain.corner(U.dot(d)) for d in ds]
		
		#Lcs = [Lhat.dot(c) for c in cs]

		# Find the unique points 
		I = unique_points(cs)
		cs = np.array(cs)[I]

		X0 = np.zeros((Nsamp, len(domain)))
		alpha = sample_simplex(len(cs), 1)
		X0[0] = cs.T.dot(alpha[0])
		for i in range(1, Nsamp):
			# Use a hackish version of Mitchel's best candidate to 
			# uniformly (ish) sample the domain
			# TODO: Would a grid of points inside simplex plus 
			alphas = sample_simplex(len(cs), 1000)
			Xcan = (cs.T.dot(alphas.T)).T
			# Find the closest point
			d = np.min(cdist(X0[0:i], Xcan), axis = 0)
			k = np.argmax(d)
			X0[i] = Xcan[k]
		return X0

def seq_maximin_sample(domain, Xhat, L = None, Nsamp = int(1e4), X0 = None):
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

	if X0 is None and L is None:
		X0 = domain.sample(Nsamp)
	elif X0 is None and L is not None:
		X0 = initial_sample(domain, L, Nsamp = Nsamp)

	Xcan = voronoi_vertex(domain, Xhat, X0, L = L, randomize = True)

	# Euclidean distance
	De = cdist(Xcan, Xhat)
	if L is not None:
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


def fill_distance_estimate(domain, Xhat, L = None, Nsamp = int(1e4), X0 = None ):
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
		 
