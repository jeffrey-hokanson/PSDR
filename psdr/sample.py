from __future__ import print_function
import numpy as np
import time
import scipy.linalg
from scipy.linalg import solve, svd
import scipy.spatial.qhull

from pool import SequentialPool
#from opt import projected_closest_point

from scipy.spatial.distance import cdist, pdist, squareform
from geometry import sample_sphere, voronoi_vertices, candidate_furthest_points, sample_boundary 
from polyridge import UnderdeterminedException, IllposedException


def maximin_sample(X, domain, L, nboundary = 500):
	"""Sequential maximin sampling using a given metric
	
	This sampling approach tries to find a new sample x 
	that maximizes the minimum distance to the already sampled points

		x = argmax_{x in domain) min_{i} || L(x - x_i)||_2

	Parameters
	----------
	X: np.array (M,m)
		Samples from domain
	domain: Domain
		Domain of dimension m on which f is posed
	L: np.array(m,m)
		Lipschitz matrix (not necessarily lower triangular
	nboundary: int
		Number of samples to take on the boundary of the domain

	Returns
	-------
	np.array(m)
		Sample solving optimization problem
	"""

	# Construct a low-rank L if possible
	U, s, VT = scipy.linalg.svd(L)
	I = np.argwhere(s> 1e-10).flatten()
	Lhat = np.diag(s[I]/s[0]).dot(VT[I,:])

	# Sample from the boundary and the largest point on the interior		
	Xcan = candidate_furthest_points(X, domain, L = Lhat, nboundary = nboundary, ninterior = 1)

	# Determine which point is actually furthest away from current points
	Ycan = np.dot(Lhat, Xcan.T).T
	Y = np.dot(Lhat, X.T).T
	dist = np.min(cdist(Ycan, Y), axis = 1)
	k = np.argmax(dist)
	return Xcan[k]

def fill_distance(X, domain, L = None, **kwargs):
	""" Computes the fill distance of a set of points

	The fill distance of a set of points X = [ x_1, x_2, ..., x_N] subset Omega
	for a bounded domain Omega is defined to be

		sup_{x in Omega} min_i || L (x - x_i) ||_2

	See [Def. 1.4, Wen04].  Here we introduce an optional scaling parameter

	This code acts as a thin wrapper to maximin_sample.

	Fill distance is also known as the "dispersion" of the set X.

	Parameters
	----------
	X: np.array
		Samples from domain
	domain: Domain
		Domain on which samples were drawn
	L: None or np.array
		Weighting on 2-norm; defaults to the identity matrix
	kwargs:
		Additional arguments passed to maximin_sample 

	Returns
	-------
	float:
		estimated fill distance
	
	Bibliography
	------------
	[Wen04] "Scattered Data Approximation", Holger Wendland, 
		Cambridge University Press, 2004.
	""" 
	if L is None:
		L = np.eye(len(domain))

	x_new = maximin_sample(X, domain, L = L, **kwargs)

	# Now compute the fill distance
	Lx_new = np.dot(L, x_new).reshape(1,len(domain))
	LX = np.dot(L, X.T).T
	fill_dist = np.min(cdist(Lx_new, LX))
	return fill_dist	
	

def multiobjective_maximin_sample(X, domain, Ls, nboundary = 500):
	""" Similar to maximin_sample, but choses based on largest distance

	"""

	if max( [L.shape[0] > 1 for L in Ls]):
		# If at least one L is not one dimensional, sample the boundary only once
		Xbndry = sample_boundary(nboundary)
		Xinterior = [maximin_sample(X, domain, L, nboundary = 0) for L in Ls] 
		Xcan = np.vstack([Xinterior, Xbndry])
	else:
		Xcan = [maximin_sample(X, domain, L, nboundary = nboundary) for L in Ls]

	ibest = []
	dist_best = []
	for k, L in enumerate(Ls):
		Y = np.dot(L, X.T).T
		Ycan = np.dot(L, Xcan.T).T
		dist = np.min(cdist(Ycan,Y), axis = 1)
		ibest.append(np.argmax(dist))
		dist_best.append(dist[ibest[-1]])

	k = np.argmax(dist_best)
	i = ibest[k] 
	return Xcan[i]



class Sampler(object):
	def __init__(self, f, domain, pool = None, X0 = None, fX0 = None):
		
		# Copy over variables
		self.f = f
		self.domain = domain
		
		if pool is None:
			pool = SequentialPool()
		self.pool = pool

		if X0 is None:
			self.X = []
		else:
			self.X = [x for x in X0]
		if fX0 is None:
			self.fX = []
		else:
			self.fX = [fx for fx in fX0]
		

	def _draw_sample(self, Xrunning):
		raise NotImplementedError

	def sample(self, draw = 1):
		""" 
		"""
		for k in range(draw):
			Xrunning = np.zeros((0, len(self.domain))) 
			xnew = self._draw_sample([Xrunning,])
			job = self.pool.apply(self.f, args = [xnew,])
			fxnew = job.output
			self.X  += [xnew]
			self.fX += [float(fxnew)]
			

	def parallel_sample(self, draw = 1, dt = 0.1):
		# TODO: Add assertion about pool support async 
		njobs = 0
		jobs = []
	
		while njobs < draw:
			# If we have a worker avalible 
			if self.pool.avail_workers() > 0:
				# Determine which jobs are done
				done = [k for k, job in enumerate(jobs) if job.ready()]

				# Get the updated information
				Xnew = [jobs[k].args[0] for k in done]
				fXnew = [jobs[k].output for k in done]

				# Update the input/ouput pairs
				self.X  += Xnew
				self.fX += fXnew

				# Delete the completed jobs
				jobs = [jobs[k] for k in enumerate(jobs) if k not in done]
				
				# Extract the points that are currently running
				Xrunning = [jobs.args[0] for k in jobs] 
				
				# Draw a sample and enqueue it
				x = self._draw_sample(Xrunning)
				jobs.append(self.pool.apply_async(self.f, args = [x,]))
				njobs += 1	
			else:
				time.sleep(dt)

		# Wait for all jobs to finish
		self.pool.join()		
				
		Xnew  = [job.args[0] for job in jobs]
		fXnew = [job.output for job in jobs]

		# Update X, fX
		self.X += Xnew
		self.fX += fXnew



class RandomSampler(Sampler):
	def _draw_sample(self, Xrunning):
		return self.domain.sample()	

class UniformSampler(Sampler):
	""" Sample uniformly on the normalized domain

	"""
	def __init__(self, f, domain, pool = None, X0 = None, fX0 = None):
		Sampler.__init__(self, f, domain, pool = pool, X0 = X0, fX0 = fX0) 		
		self.domain_norm = self.domain.normalized_domain()
		self.L = np.eye(len(self.domain))

	def _draw_sample(self, Xrunning):
		Xall = np.array(self.X + Xrunning)
		Xall_norm = self.domain.normalize(Xall)
		
		xnew_norm = maximin_sample(Xall_norm, self.domain_norm, self.L)
		xnew = self.domain.unnormalize(xnew_norm)
		return xnew

class RidgeSampler(Sampler):
	"""

	Note: the ridge approximation is always formed on the normalized domain

	Parameters
	----------
	pra: Instance of PolynomialRidgeApproximation
	"""
	def __init__(self, f, domain, pra, **kwargs):
		Sampler.__init__(self, f, domain, **kwargs)
		self.domain_norm = self.domain.normalized_domain()
		self.pra = pra
		self.U = None
		self.fill_dist = np.inf

	def _draw_sample(self, Xrunning):
		if len(self.fX) <= len(self.domain)+1:
			return self.domain.sample()

		# Build ridge approximation
		X_norm = self.domain.normalize(self.X)
		fX = np.array(self.fX)
		I = np.isfinite(fX)
		try:
			self.pra.fit(X_norm[I], fX[I])
		except (UnderdeterminedException, IllposedException):
			# If we can't yet solve the problem, sample randomly
			return self.domain.sample()		

		Xall = np.vstack(self.X + Xrunning)
		Xall_norm = self.domain.normalize(Xall)
		self.U = self.pra.U
		xnew_norm = maximin_sample(Xall_norm, self.domain_norm, L = self.U.T)
		self.fill_dist = np.min(cdist(self.U.T.dot(xnew_norm).reshape(1,-1), self.U.T.dot(Xall_norm.T).T))
		xnew = self.domain.unnormalize(xnew_norm)
		return xnew
					
if __name__ == '__main__':
	from demos import golinski_volume, build_golinski_design_domain
	from poly_ridge import PolynomialRidgeApproximation
	dom = build_golinski_design_domain()
		
	f = golinski_volume
	
	np.random.seed(0)

	pra = PolynomialRidgeApproximation(degree = 5, subspace_dimension = 1)
	samp = RidgeSampler(f, dom, pra)
	samp.sample(2)
	for k in range(2,1000):
		samp.sample()
		print("%3d %5.2e" % (k, samp.fill_dist))
		 
