from __future__ import print_function
import numpy as np
from scipy.spatial.distance import cdist

from .vertex import voronoi_vertex 


__all__ = ['seq_maximin_sample', 'fill_distance_estimate' ]

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

	if X0 is None:
		X0 = domain.sample(Nsamp)

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

	The *fill distance* (Def. 1.4 of [Wen04]_) or *dispersion*
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
	"""
	
	if X0 is None:
		X0 = domain.sample(Nsamp)

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
		 
