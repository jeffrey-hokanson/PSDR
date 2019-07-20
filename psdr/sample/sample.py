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


try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache


from ..geometry import voronoi_vertex 
from ..geometry import sample_sphere, unique_points, sample_simplex
from ..domains import LinIneqDomain, ConvexHullDomain

from .initial import initial_sample

__all__ = ['seq_maximin_sample', 'Sampler', 'SequentialMaximinSampler',
	'StretchedSampler', 'lipschitz_sample']



def lipschitz_sample(domain, Nsamp, Ls, maxiter = 100, verbose = False, jiggle = False):
	r""" Construct a maximin design with respect to multiple Lipschitz matrices




	Parameters
	----------
	domain: Domain
		domain from which the samples are taken
	Nsamp: int
		Number of points to take
	Ls: list of arrays of size (*,m)
		Weighted metrics
	maxiter: int, optional
		Number of feasible designs that are randomly selected from
		the space of available solutions.
	verbose: bool, optional
		If True, print iteration information
	jiggle: bool or int, optional
		If True or an int, draw that number of samples from each sub-domain
		and test for optimality
	"""	 
	
	if jiggle is True:
		jiggle = 20


	Lall = np.vstack(Ls)
	
	# Construct maximin points for each metric L
	ys = []
	As = [ [None for j in range(Nsamp)] for L in Ls]
	bs = [ [None for j in range(Nsamp)] for L in Ls]
	for i, L in enumerate(Ls):
		X = maximin_sample(domain, Nsamp, L)
		y = L.dot(X.T).T
		ys.append(y)

		# Construct the set of neighbors for each point
		if y.shape[1] == 1:
			# Since Qhull doesn't support 1-d, we explicitly compute the neighbors
			# for the 1-d case.
			I = np.argsort(y.flatten())
			neighbors = [ None for j in range(Nsamp)]
			for j in range(Nsamp):
				if j == 0:
					neighbors[I[j]] = np.array([I[j+1]], dtype = np.int)
				elif j == Nsamp - 1:
					neighbors[I[j]] = np.array([I[j-1]], dtype = np.int)
				else:
					neighbors[I[j]] = np.array([ I[j-1], I[j+1] ], dtype = np.int)
			
		else:
			delaunay = Delaunay(y)
			indices, indptr = delaunay.vertex_neighbor_vertices
			# Convert the Delaunay data structure into a list of neighbors
			# https://stackoverflow.com/a/23700182
			neighbors = [indptr[indices[k]:indices[k+1]] for k in range(len(y))]
		
		# Now construct the inequality constraints for each point's Voronoi cell
		for j, neigh in enumerate(neighbors):
			A = []
			b = []
			for k in neigh:
				p = 0.5*(X[j] + X[k]) 	# point on separating hyperplane
				n = (X[k] - X[j]).reshape(-1,1) 		# normal vector to hyperplane
				A.append( n.T.dot(L.T).dot(L))
				b.append( n.T.dot(L.T).dot(L).dot(p) )
			
			As[i][j] = np.vstack(A)
			bs[i][j] = np.hstack(b)
			dom = domain.add_constraints(A = As[i][j], b = bs[i][j])
			assert dom.isinside(X[j])
			assert np.sum(dom.isinside(X)) == 1	


	def encode(metric, order, value):
		# Python3 needs this explicitly an int to interface with pylgl
		return int(metric*Nsamp*Nsamp + order*Nsamp + value + 1)
	def decode(idx):
		return (idx-1) // Nsamp**2, ((idx-1) % Nsamp**2)// Nsamp, (idx-1) % Nsamp

	@lru_cache(maxsize = int(1e6) )
	def subdomain(idx):
		A = np.vstack([As[metric][idx[metric]] for metric in range(len(Ls))])
		b = np.hstack([bs[metric][idx[metric]] for metric in range(len(Ls))])
		return domain.add_constraints(A = A, b = b)
	
	@lru_cache(maxsize = int(1e6))
	def subdomain_sample(idx):
		# The point in the domain that is nearest to the target position
		subdom = subdomain(idx)
		A = np.vstack([Ls])
		b = np.hstack([y[i] for y, i in zip(ys, idx)])
		return subdom.constrained_least_squares(A, b)
			

	# Construct the geometric constraints we enforce with the SAT solver
	# i.e., no repetition	
	geo_cnf = []
	for metric in range(len(Ls)):
		for order in range(Nsamp):
			# At least one must be one
			geo_cnf += [ [encode(metric, order, value) for value in range(Nsamp)] ]	
			# Not more than one is on
			# we do this by checking not ( v1 and v2) which is equivalent to 
			# not v1 or not v2 by DeMorgan
			geo_cnf += [ [-encode(metric, order, v1), -encode(metric, order, v2)] 
						for v1, v2 in zip(*np.triu_indices(Nsamp,1)) ]

		# Each equality constraint cannot be selected more than once
		for value in range(Nsamp):
			geo_cnf += [ [-encode(metric, o1, value), -encode(metric, o2, value)] 
						for o1, o2 in zip(*np.triu_indices(Nsamp, 1)) ]

	# To fix ordering of the samples, we fix the ordering of the first metric
	geo_cnf += [ [encode(0, value, value)]  for value in range(Nsamp)]


	score_best = tuple([0. for i in range(Nsamp)])
	perms_best = []
	X_best = None

	# Initialize the solver and tie its random seed to numpy's random state
	sat_iter = picosat.itersolve(geo_cnf, initialization = 'random', seed = np.random.randint(2**16))

	it = 0
	while True:
		try:
			sol = sat_iter.next()
		except (KeyboardInterrupt, SystemExit):
			raise
		except StopIteration:
			try:
				# This automatically releases previous assumptions
				if verbose: print("assumptions failed")
				sol = sat_iter.next()
			except (KeyboardInterrupt, SystemExit):
				raise
			except:
				break
	
		# Convert the solution format into a permutation matrix
		sol = np.array(sol)
		perms = -np.ones((len(Ls), Nsamp), dtype = np.int)
		for i in np.argwhere(sol > 0).flatten():
			metric, order, value = decode(sol[i])
			perms[metric, order] = value

		# Construct a series of domains from this set of constraints
		subdoms = []
		for order in range(Nsamp):
			subdoms.append(subdomain(tuple(perms[:,order])))
	
		# Find domains that are empty and remove that combination
		# This happens here so we don't need to iterate over all combinations
		# at the start---something that is expensive in cases where only a small 
		# fraction of subdomains are invalid
		new_geo_cnf = []
		for order, subdom in enumerate(subdoms):
			if subdom.is_empty:
				# If the subdomain is empty, block future samples from using this one
				new_geo_cnf += [ [-encode(metric, o, perms[metric, order]) for metric in range(len(Ls))] 
						for o in range(Nsamp)] 

		sat_iter.add_clauses(new_geo_cnf)
		geo_cnf += new_geo_cnf

		if all([ not subdom.is_empty for subdom in subdoms]):
			print(perms[1:,:])
			it += 1	

	
			X = np.vstack([subdomain_sample(tuple(perms[:,order])) for order in range(Nsamp)])
			
			if Lall.shape[0] < len(domain):
				# If the Ls do not completely fix each of the samples, iterate to find maximin design
				consubdoms = [subdom.add_constraints(A_eq = Lall, b_eq = Lall.dot(x)) for subdom, x in zip(subdoms, X)]
				X = np.vstack([consubdom.sample() for consubdom in consubdoms])
				mask = np.ones(Nsamp, dtype = np.bool)
				for it2 in range(100):
					max_move = 0
					for i in range(Nsamp):
						# Remove the current iterate 
						mask[i] = False
						Xt = X[mask,:]
						# Reset the mask
						mask[i] = True 
						x = voronoi_vertex(consubdoms[i], Xt, X[i])	
	
						# Compute movement of this point
						move = np.linalg.norm(X[i] - x.flatten(), np.inf)
						max_move = max(max_move, move)
						
						# update this point
						X[i] = x
					if max_move < 1e-6:
						break
			
			D = squareform(pdist(X))
			D += np.max(D)*np.eye(D.shape[0])
			min_dist = np.min(D, axis = 0)
			score = tuple(np.sort(min_dist))
			
			if score > score_best:
				X_best = X
				score_best = score
				perms_best = perms
				updated = True
			
				# Force solver to keep best
				keep = np.argsort(-min_dist)[0:-len(Ls)*2]
				assumptions = []
				for k in keep:
					assumptions += [encode(metric, k, perms[metric,k]) for metric in range(len(Ls))]
			else: 
				updated = False

			sat_iter.assume(assumptions)
			
			if verbose:
				mess = "it %4d:" % (it,) 
				mess += 'best: '
				mess += ' '.join(['%3.2f' % s for s in score_best])
				if updated:
					mess += ' *updated*'
				print(mess)
		else:
			if verbose:
				print('found invalid domain')
		if it >= maxiter:
			break
	return X_best 



def seq_maximin_sample(domain, Xhat, Ls = None, Nsamp = int(1e3), X0 = None, slack = 0.9):
	r""" A multi-objective sequential maximin sampling 
	

	Given an existing set of samples :math:`\lbrace \widehat{\mathbf{x}}_j\rbrace_{j=1}^M\subset \mathcal{D}`
	from the domain :math:`\mathcal{D} \subset \mathbb{R}^m`, this algorithm finds a point :math:`\mathbf{x} \in \mathcal{D}`
	that approximately maximizes the distance of the new point to all other points in multiple distance metrics 
	give by matrices :math:`\mathbf{L}_i \in \mathbb{R}^{m\times m}`

	.. math::

		\max_{\mathbf{x} \in \mathcal{D}} \left\lbrace 
			\min_{j=1,\ldots,M} \|\mathbf{L}_i (\mathbf{x} - \widehat{\mathbf{x}}_j)\|_2
			\right\rbrace_{i}

	This algorithm uses :meth:`psdr.voronoi_vertex` to generate local maximizers of this problem
	for each metric and then tries to greedily satisfy the distance requirements for each metric.

	A typical use case will have Ls that are of size (1,m)	
	This greedy sequential approach for constructing a maximin design is 
	the Coffee-House Designs of Muller [Mul01]_. However, the approach of Muller
	allows for a generic nonlinear solve for each sample point.  Here though
	we restrict the domain to a polytope specified by linear inequalities
	so we can invoke :meth:`psdr.voronoi_vertex` to solve each step. 

	Parameters
	----------
	domain: Domain
		The domain from which we will be sampling
	Xhat: array-like (M, m)
		Previously existing samples from the domain 
	Ls: list of array-like (?, m) matrices, optional
		The weight matrix (e.g., Lipschitz matrix) corresponding to each metric;
		defaults to the identity matrix
	Nsamp: int, optional (default 1000)
		Number of samples to use when finding Voronoi vertices
	slack: float [0,1], optional (default 0.1)
		Rather than taking the point that maximizes the product of the
		distances in each metric, we choose the point x with greatest unweighted Euclidean
		distance from those candidates that are at least slack times the score of the best.

	References
	----------
	.. [Mul01] Coffee-House Designs.
		Werner G. Muller
		in Optimimum Design 2000, A. Atkinson et al. eds., 2001
	"""
	if Ls is None:
		Ls = [np.eye(len(domain))]

	# If we have no samples we pick a corner in the direction
	# of the dominant singular vector of the stacked L matrices
	if len(Xhat) == 0:
		Lall = np.vstack(Ls)
		_, s, VT = scipy.linalg.svd(Lall)
		# If several singular values are close, we randomly select a direction 
		# from that subspace
		I = np.argwhere(np.isclose(s, s[0])).flatten()
		u = VT.T[:,I].dot(np.random.randn(len(I)))
		return domain.corner(u)
	
	Xhat = np.array(Xhat)
	Xhat = np.atleast_2d(Xhat)

	#############################################################################
	# Otherwise, we proceed with identifiying Voronoi vertices associated with
	# each of the metrics (L's) provided.
	#############################################################################

	vertices = []
	distances = []
	for k, L in enumerate(Ls):
		# Find initial samples well separated
		if X0 is None:
			X = initial_sample(domain, L, Nsamp = Nsamp//(len(Ls)+1))
		else:
			X = np.copy(X0)

		# find the Voronoi vertices; we don't randomize as we are only interested
		# in the component that satisfies the constraint
		vert = voronoi_vertex(domain, Xhat, X, L = L, randomize = False) 
		
		# Remove duplicates in the L norm
		I = unique_points(L.dot(vert.T).T)
		vert = vert[I]

		# Compute the distances between points in this metric
		D = cdist(L.dot(vert.T).T, L.dot(Xhat.T).T)
		D = np.min(D, axis = 1)
		# Order the vertices in decreasing distance
		I = np.argsort(-D)
		vert = vert[I]
		vertices.append(vert)
		distances.append(D[I])
		
	#############################################################################
	# Now we construct a number of candidate domains to sample from
	# many of these may be empty because the constraints are infeasible
	#############################################################################

	# When generating these domains, we limit the number of vertices we consider
	max_verts = max(2, int(np.floor(1e2**(1./len(Ls) ))))

	# A list of which vertices to consider at each step
	coords = []
	for dist, vert in zip(distances, vertices):
		#if dist[0] == np.max([d[0] for d in distances]):
		#	# If this coordinate has the largest distance, we only sample the largest one 
		#	coords.append([0])
		#else:
		# Otherwise we sample the first few largest
		coords.append(np.arange(min(len(vert),max_verts)))

	# Generate a score associated with each
	# This score is the product to the distances in each metric 
	idx = list(itertools.product(*coords))
	dist_prod = [ sum([np.log10(dist[i]) for dist, i in zip(distances, idx_i)]) for idx_i in idx]

	# Order these in decreasing score
	I = np.argsort(-np.array(dist_prod))
	idx = [idx[i] for i in I]

	Xcan = []
	score_Ls = []
	used_idx = []
	for it in range(100):
		new_domain = False
		while len(idx) > 0:
			# Grab a combination of constraints to try
			idx_i = idx.pop(0)

			# These are the used indices; negative meaning no constraint applied
			found_idx = -1*np.ones(len(idx_i))

			domain_samp = domain	
			# Add the constraints on iteratively in decreasing distance
			for k in np.argsort([-dist[i] for i, dist in zip(idx_i, distances)]):
				L = Ls[k]
				vert = vertices[k][idx_i[k]]
				domain_test = domain_samp.add_constraints(A_eq = L, b_eq = L.dot(vert) )
				if domain_test.is_empty:
					#print("empty after %d constraints" % k)
					break
				else:
					domain_samp = domain_test
					found_idx[k] = idx_i[k]

			#print('found_idx', found_idx)
			#if found_idx not in used_idx:
			if len(used_idx) == 0 or np.min([np.linalg.norm(found_idx - used_idx_i) for used_idx_i in used_idx]) > 0:
				used_idx.append(found_idx)
				new_domain = True
				break

		if not new_domain:
			break
		
		# Generate candidates 
		X0 = initial_sample(domain_samp, np.eye(len(domain)), Nsamp = 100)
		Xcan_new = voronoi_vertex(domain_samp, Xhat, X0)
		
		# Score samples: product of distances in each of the L metrics
		score_Ls_new = np.ones(Xcan_new.shape[0])
		for L in Ls:
			D = cdist(L.dot(Xcan_new.T).T, L.dot(Xhat.T).T)
			d = np.min(D, axis = 1)
			with np.errstate(divide='ignore'):
				score_Ls_new *= d # np.log10(d)
		score_Ls = np.hstack([score_Ls, score_Ls_new])
		Xcan.append(Xcan_new)
		
		# If remaining candidates are too close, break
		# (and we've used all the constraints)
		#print("score", np.max(score_Ls_new), "best", np.max(score_Ls), "b_eq", domain_samp.b_eq, "idx_i", idx_i)
		active_constraints = np.sum(found_idx >=0)
		if np.max(score_Ls_new) < slack*np.max(score_Ls) and active_constraints == len(Ls):
			# This prevents us from generating candidates that will be removed 
			#print("stopping")
			break

	#print("done with sampling")
	
	Xcan = np.vstack(Xcan)
	# Remove duplicates
	I = unique_points(Xcan)
	Xcan = Xcan[I]	
	score_Ls = score_Ls[I]
	
	
	# Compute Euclidean distances	
	D = cdist(Xcan, Xhat)
	score_I = np.min(D, axis = 1)
	
#	for i in np.argsort(-score_Ls):
#		print("%3d\t %g \t %g" % (i, score_Ls[i], score_I[i]))

#	import matplotlib.pyplot as plt
#	fig, ax = plt.subplots()
#	ax.plot(score_Ls, score_I, 'k.')
#	ax.set_xlabel('Ls score')
#	ax.set_ylabel('I score')
#	plt.show()

	# Now select the one within 95% of optimum 
	I = (score_Ls >= np.max(score_Ls)*slack)
	# Delete those not matching critera
	Xcan = Xcan[I]
	score_I = score_I[I]
	# pick the remaining point with highest Euclidean metric
	i = np.argmax(score_I)
	return Xcan[i]


	
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
		 