import numpy as np
import time
from scipy.linalg import solve, svd
import scipy.spatial.qhull

from pool import SequentialPool
from opt import projected_closest_point

from scipy.spatial.distance import cdist, pdist, squareform
from geometry import sample_sphere, voronoi_vertices 


def maximin_sample(X, domain, L, nboundary = 500, ncorner = 100):
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

	# See if L reduces the dimension
	U, s, VT = svd(L)
	V = VT.T
	s = s/s[0]
	L_norm = L/s[0]
	I = (s>1e-14)	
	sVT = np.dot(np.diag(s[I]), VT[I,:])
	Y = np.dot(sVT, X.T).T	

	# Attempt to sample the interior, reducing dimension if necessary
	while True:
		try:
			# Compute interior points
			# TODO: also switch to random sampling if dimension is too large
			Yinterior = voronoi_vertices(Y)	
			break
		except scipy.spatial.qhull.QhullError:
			# If we don't have enough points, we need to reduce the effective dimension of Y
			I[np.sum(I)-1] = False
			sVT = np.dot(np.diag(s[I]), VT[I,:])
			Y = np.dot(sVT, X.T).T	

	n = Y.shape[1]
						
	# Sample on the boundary
	if n == 1:
		# If we are one-dimensional, we only need to probe the corners
		# TODO: Check this direction is right
		Xbndry = [domain.corner(V[:,0]), domain.corner(-V[:,0])]
	else:
		center = domain.center
		Z = sample_sphere(len(domain), nboundary)
		# TODO: are these the right directions to sample?
		#ZZ = [np.dot(V ,np.dot(np.diag(s), np.dot(U[I,:].T, z))) for z in Z]
		#ZZ = [np.dot(L_norm.T, z) for z in Z]
		Xbndry = [center + zz*domain.extent(center, zz) for zz in Z]
		#print [domain.extent(center, z) for z in VZ]
		#print domain.isinside(np.array(Xbndry))
		Z = sample_sphere(len(domain), ncorner)
		#ZZ = [np.dot(V ,np.dot(np.diag(s), np.dot(U[I,:].T, z))) for z in Z]
		#ZZ = [np.dot(L_norm.T, z) for z in Z]
		Xbndry += [domain.corner(zz) for zz in Z]

	Xbndry = np.array(Xbndry)
	Ybndry = np.dot( sVT, Xbndry.T).T
	
	# Now compute distance from our candidates to the existing points
	Yall = np.vstack([Yinterior, Ybndry])
	D = cdist(Yall, Y)
	min_dist = np.min(D,axis = 1)

	while np.max(min_dist) > 0:
		k = np.argmax(min_dist)
		#print "distance %5.2e" % (min_dist[k],)
		if k < len(Yinterior):
			y = Yinterior[k]
			try:
				dom_con = domain.add_constraint(A_eq = sVT, b_eq = y)
			except InfeasibleConstraints:
				#print "infeasible"
				min_dist[k] = -1
			else:
				Xnew = np.array([dom_con.corner(np.random.randn(len(dom_con))) for it in range(10)])
				k2 = np.argmin(np.min(cdist(Xnew,X), axis = 1)) 
				x = Xnew[k2]
				break
		else:
			# Othwerise we've sampled on the boundary
			k -= len(Yinterior)
			x = Xbndry[k]
			break
	#print "distance of furthest point", np.min(cdist( x.reshape(1,-1), X) ), np.min(cdist( np.dot(sVT,x).reshape(1,-1), Y) )
	return x


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

	def sample(self, draw = 1, dt = 0.1):
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
				jobs.append(self.pool.apply(self.f, args = [x,]))
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

class LipschitzSampler(Sampler):
	def __init__(self, f, domain, L_norm, pool = None, X0 = None, fX0 = None):
		Sampler.__init__(self, f, domain, pool = pool, X0 = X0, fX0 = fX0) 		
		self.L_norm = L_norm
		self.domain_norm = self.domain.normalized_domain()

	def _draw_sample(self, Xrunning):
		Xall = np.array(self.X + Xrunning)
		Xall_norm = self.domain.normalize(Xall)
		xnew_norm = maximin_sample(Xall_norm, self.domain_norm, self.L_norm)
		xnew = self.domain.unnormalize(xnew_norm)
		return xnew

class SequentialLipschitzSampler(Sampler):
	""" Sequential Lipschitz Sampling

	Samples points uniformly based on Lipschitz matrix,
	updating this Lipschitz matrix based on the samples.

	"""

	def __init__(self, *args, **kwargs):
		Sampler.__init__(self, *args, **kwargs)
		self.L_norm = np.zeros((len(self.domain), len(self.domain)))
		self.domain_norm = self.domain.normalized_domain()

	def _draw_sample(self, Xrunning):

		# determine if we need to update the Lipschitz matrix
		X = np.array(self.X)
		X_norm = self.domain.normalize(X)
		fX = np.array(self.fX)
		if len(X) > 2:
			if check_lipschitz(self.L_norm, X = X_norm, fX = fX) < 0:
				self.L_norm = multivariate_lipschitz(X = X_norm, fX = fX)	
			Xall = np.array(self.X + Xrunning)
			Xall_norm = self.domain.normalize(Xall)
			xnew_norm = maximin_sample(Xall_norm, self.domain_norm, self.L_norm)
			xnew = self.domain.unnormalize(xnew_norm)
		else:
			xnew = self.domain.sample()

		return xnew	

					
if __name__ == '__main__':
	from psdr.demos import golinski_volume, build_golinski_design_domain
	dom = build_golinski_design_domain()
	
	f = golinski_volume
	
	np.random.seed(0)

	samp = LipschitzSampler(f, dom)
	samp.sample(2)
	for k in range(2,20):
		print "==========  %3d =============" % (k,)
		samp.sample()
		print samp.L
		#print samp.X
		#print samp.fX
	#X = np.copy(samp.X)
	#L = samp.L
	#for k in range(10):
	#	xnew = maximin_sample(X, dom, L, nboundary = 1000)
	#	X = np.vstack([X, xnew.reshape(1,-1)])
		 
