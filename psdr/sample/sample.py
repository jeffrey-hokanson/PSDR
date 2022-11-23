from __future__ import print_function
import numpy as np
import numpy.ma as ma
import scipy.linalg
import scipy.optimize
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import Delaunay
import cvxpy as cp
import itertools

from satyrn import picosat


from ..geometry import voronoi_vertex_sample 
from ..geometry import sample_sphere, unique_points, sample_simplex
from ..domains import LinIneqDomain, ConvexHullDomain

from .initial import initial_sample

__all__ = ['Sampler', 'SequentialMaximinSampler', 'StretchedSampler']




	
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
			xnew = seq_maximin_sample(self._fun.domain, self._X, Ls = [self._L])
			Xnew.append(xnew)
			if verbose:
				print('%3d: ' % (i,),  ' '.join(['%8.3f' % x for x in xnew]))
			self._X = np.vstack([self._X, xnew])

		# Now we evaluate the function at these new points
		# (this takes advantage of potential vectorization of fun)
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
		 
