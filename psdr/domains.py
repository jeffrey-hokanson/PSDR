"""Base domain types"""

import numpy as np
from scipy.optimize import newton, brentq
from copy import deepcopy

from scipy.optimize import nnls, minimize
from scipy.linalg import orth
from scipy.spatial import ConvexHull

from util import *

#__all__ = ['Domain', 'ComboDomain', 'BoxDomain', 'UniformDomain', 'NormalDomain', 'LogNormalDomain', 'LinIneqDomain', 'ConvexHullDomain'] 


class DomainException(Exception):
	pass

class UnboundedDomain(DomainException):
	pass

class EmptyDomain(DomainException):
	pass


def auto_root(dist): 
	# construct initial bracket
	a = 0.0
	b = 1.0
	for i in range(1000):
		dist_b = dist(b)
		#print 'bracket:', a, b, 'dist', dist_b
		if np.isfinite(dist_b) and dist_b > 0:
			break
		if not np.isfinite(dist_b):
			b = (a + b)/2.0
		if dist_b < 0:
			a = b
			b *= 2.0 
	alpha_brent = brentq(dist, a, b, xtol = 1e-12, maxiter = 100)
	return alpha_brent


class Domain(object):
	""" Abstract base class for an input domain
	"""
	# To define the documentation once for all domains, these functions call internal functions
	# to each subclass
	def sample(self, draw = 1):
		""" Sample points from the domain according to its measure

		Parameters
		----------
		draw: int
			Number of samples to return

		Returns
		-------
		np.ndarray (draw, len(self))
			Array of samples from the domain
		"""
		x_sample = self._sample(draw = int(draw))
		if draw == 1: 
			x_sample = x_sample.flatten()
		return x_sample


	def extent(self, x, p):
		"""Compute the distance alpha such that x + alpha * p is on the boundary of the domain

		Parameters
		----------
		x : np.ndarray(m)
			Starting point in the domain

		p : np.ndarray(m)
			Direction from p in which to head towards the boundary

		Returns
		-------
		alpha: float
			Distance to boundary along direction p
		"""
		assert self.isinside(x), "Starting point must be inside the domain" 
		return self._extent(x, p)

	def isinside(self, X):
		""" Determine if points are inside the domain

		Parameters
		----------
		X : np.ndarray(M, m)
			Samples in rows of X
		"""
		# Make this a 2-d array
		if len(X.shape) == 1:
			X = X.reshape(-1, len(self)) 	
			return self._isinside(X).flatten()
		else:
			return self._isinside(X)


	def corner(self, p):
		""" Find the point furthest in direction p inside the domain

		Solves the optimization problem
	
			max  p^T x
			x in domain

		Parameters
		----------
		p: np.ndarray(m,)
			direction in which to maximize
		"""
		return self._corner(p)

	def normalize(self, X):
		""" Given a points in the application space, convert it to normalized units
		
		Parameters
		----------
		X: np.ndarray((M,m))
			points in the domain to normalize
		"""
		if len(X.shape) == 1:
			X = X.reshape(-1, len(self)) 
			return self._normalize(X).flatten()
		else:
			return self._normalize(X)

	def unnormalize(self, X_norm):
		""" Convert points from normalized units into application units
		
		Parameters
		----------
		X_norm: np.ndarray((M,m))
			points in the normalized domain to convert to the application domain
		
		"""
		if len(X_norm.shape) == 1:
			X_norm = X_norm.reshape(-1, len(self)) 
			return self._unnormalize(X_norm).flatten()
		else:
			return self._unnormalize(X_norm)
	
	def normalized_domain(self):
		""" Return a domain with units normalized corresponding to this domain
		"""
		return self._normalized_domain()
	
	def __add__(self, other):
		""" Combine two domains
		"""
		assert isinstance(other, Domain)
		if isinstance(other, ComboDomain):
			ret = deepcopy(other)
			ret.domains.insert(0, deepcopy(self))
		else:
			ret = ComboDomain()
			ret.domains = [deepcopy(self), deepcopy(other)]
		return ret
	
	def __radd__(self, other):
		""" Combine two domains
		"""
		assert isinstance(other, Domain)
		if isinstance(other, ComboDomain):
			ret = deepcopy(other)
			ret.domains.append(deepcopy(self))
		else:
			ret = ComboDomain()
			ret.domains = [deepcopy(other), deepcopy(self)]
		return ret
	

	def projected_closest_point(self, A, b):
		""" Computes the closest projected point, solving

			min_{x in domain} \| A x - b\|_2^2
			
		""" 
		if isinstance(self, ComboDomain):
			if all([isinstance(dom, LinIneqDomain) for dom in self.domains]):
				x = projected_closest_point(A, b, A_ub = self.A, b_ub = self.b, A_eq = self.A_eq, b_eq = self.b_eq, lb = self.lb, ub = self.ub)
			else:
				raise NotImplementedError
		elif isinstance(self, LinIneqDomain):
			x = projected_closest_point(A, b, A_ub = self.A, b_ub = self.b, A_eq = self.A_eq, b_eq = self.b_eq, lb = self.lb, ub = self.ub)
		else:
			raise NotImplementedError
		
		assert self.isinside(x), "projected closest point not inside domain"
		return x
	
	def closest_point(self, x0, L = None):
		""" Find the closest point in the domain to x0

		Solve the minimization problem

			min_{x \in domain} \|L(x - x0)\|_2
		

		Parameters
		----------
		x0: np.ndarray(m)
			point to be closest to
		L: np.ndarray(m,m)
			Cholesky factorization of mass matrix
		"""
		if isinstance(self, ComboDomain):
			if all([isinstance(dom, LinIneqDomain) for dom in self.domains]):
				return closest_point(x0, L = L, A_ub = self.A, b_ub = self.b, A_eq = self.A_eq, b_eq = self.b_eq, lb = self.lb, ub = self.ub)
			else:
				raise NotImplementedError
		elif isinstance(self, LinIneqDomain):
			return closest_point(x0, L = L, A_ub = self.A, b_ub = self.b, A_eq = self.A_eq, b_eq = self.b_eq, lb = self.lb, ub = self.ub)
		else:
			raise NotImplementedError
			

class LinIneqDomain(Domain):
	""" A domain that is specified by linear inequality constraints

	Here we create a domain for points x that satisfy the following set of constraints

	Bound constraints     : lb <= x <= ub
	Inequality constraints: A x <= b
	Equality constraints  : A_eq x = b_eq

	Raises
	------
	EmptyDomain

	"""
	def __init__(self, A = None, b = None, lb = None, ub = None, A_eq = None, b_eq = None, center = None):
		if (A is None and b is None) and (lb is None and ub is None):
			raise ValueError('Either A and b must be specified or bounds lb and ub')

		if (A is None and b is not None) or (A is not None and b is None):
			raise ValueError('Both A and b must be either defined or undefined')

		if (A_eq is None and b_eq is not None) or (A_eq is not None and b_eq is None):
			raise ValueError('Both A_eq and b_eq must be either defined or undefined')
		

		# Default values
		self._radius = None
		self._z0 = None 

		# Determine the dimension of the parameter space
		if A is not None:
			m = A.shape[1]
		elif lb is not None:
			m = lb.shape[0]

		# Check dimensions of the constraints match
		if A is not None:
			assert A.shape[1] == m
			assert A.shape[0] == b.shape[0]
			assert len(b.shape) == 1
		if lb is not None:
			assert lb.shape[0] == m
		if ub is not None:
			assert ub.shape[0] == m
		if A_eq is not None:
			assert A_eq.shape[1] == m
			assert A_eq.shape[0] == b_eq.shape[0]
			assert len(b_eq.shape) == 1
		
		# Copy over the constraints
		if A is None and b is None:
			self._A = np.zeros((0, m))
			self._b = np.zeros((0,))
		else:
			self._A = np.copy(A)
			self._b = np.copy(b)

		if lb is None:
			self._lb = -np.inf*np.ones((m,))
		else:
			self._lb = np.copy(lb)
		if ub is None:
			self._ub = np.inf*np.ones((m,))	
		else:
			self._ub = np.copy(ub)
		
		if A_eq is None and b_eq is None:
			self._A_eq = np.zeros((0, m))
			self._b_eq = np.zeros((0,))
			self._A_eq_basis = np.zeros((m,0))
		else:
			self._A_eq = np.copy(A_eq)
			self._b_eq = np.copy(b_eq)
			if A_eq.shape[0] > 0:
				self._A_eq_basis = orth(A_eq.T)
			else:
				self._A_eq_basis = np.copy(A_eq.T)

		# Before continuing, check that the constraints provide a non-empty domain
		try:
			c = np.zeros((m,))
			linprog(c, A_ub = self.A, b_ub = self.b, A_eq = self.A_eq, b_eq = self.b_eq, lb = self.lb, ub = self.ub)
		except LinProgException:		
			raise EmptyDomain

		

		# If we are not provided a center, use the Chebeychev center
		if center is None:
			center, radius = self._chebyshev_center()
			self.center = center
		else:
			# TODO: This should be a try-catch statement that will except if we cannot solve QP
			#self.center = self.closest_point(center)
			self.center = center
		
		self._z0 = np.copy(center)

		# Now see if this domain contains more than a point
		if self.radius > 1e-10:
			self._can_sample = True
		else:
			# Test if there are any directions to go from the center that are not empty
			self._can_sample = False
			for it in range(10*len(self)):
				# Choose a random direction
				p = np.random.normal(size = self._z0.shape)
				p /= np.linalg.norm(p)
				
				# Project the direction onto the orthogonal complement of the 
				# range of A_eq.T, so that the step continues to satisfy the 
				# equality constraint
				p -= np.dot(self._A_eq_basis, np.dot(self._A_eq_basis.T, p))
				# Determine the extent
				alpha_min = -self.extent(self.center, -p)
				alpha_max = self.extent(self.center, p)
				if alpha_max - alpha_min > 1e-7:
					self._can_sample = True
					break
			
					


		# Now construct finite lower/upper bounds for normalization purposes
		for i in range(len(self)):
			if not np.isfinite(self.lb[i]):
				c = np.zeros((m,))
				c[i] = 1
				x = linprog(c, A_ub = self.A, b_ub = self.b, A_eq = self.A_eq, b_eq = self.b_eq)
				self._lb[i] = x[i]
	
			if not np.isfinite(self.ub[i]):
				c = np.zeros((m,))
				c[i] = -1
				x = linprog(c, A_ub = self.A, b_ub = self.b, A_eq = self.A_eq, b_eq = self.b_eq)
				self._ub[i] = x[i]
 
	def add_constraint(self, A = None, b = None, lb = None, ub = None, A_eq = None, b_eq = None, center = None):
		""" Add a constraint(s) to the domain, returning a new domain

		The inequality (A, b) and equality constraints (A_eq, b_eq) are compounded,
		whereas the bound constraints (lb, ub) are replaced.
		"""

		# Inequality constraints
		if A is not None and b is not None:
			assert A.shape[0] == b.shape[0], "both A and b must have the same number of rows"
			assert A.shape[1] == len(self), "A must have the same dimensions as the current A"
			A = np.vstack([self.A, A])
			b = np.hstack([self.b, b])
		elif b is None and A is None:
			A = self.A
			b = self.b
		else:
			raise ValueError("Both A and be must be defined")


		# Bound constraints
		if lb is not None:
			assert lb.shape[0] == len(self), "lb must have the same dimension as the current domain"
		else:
			lb = self.lb
		if ub is not None:
			assert ub.shape[0] == len(self), "lb must have the same dimension as the current domain"
		else:
			ub = self.ub
		
		# Equality constraints
		if A_eq is not None and b_eq is not None:
			assert A_eq.shape[0] == b_eq.shape[0], "both A_eq and b_eq must have the same number of rows"
			assert A_eq.shape[1] == len(self), "A_eq must have the same number of columns as the dimension of the domain"
			A_eq = np.vstack([self.A_eq, A_eq])
			b_eq = np.hstack([self.b_eq, b_eq])
		elif b_eq is None and A_eq is None:
			A_eq = self.A_eq
			b_eq = self.b_eq
		else:
			raise ValueError("Both A_eq and b_eq must be defined")
	
		#if center is None:
		#	center = closest_point(self.center, A_ub = A, b_ub = b, lb = lb, ub = ub, A_eq = A_eq, b_eq = b_eq) 
	
		return LinIneqDomain(A = A, b = b, lb = lb, ub = ub, A_eq = A_eq, b_eq = b_eq, center = center)	
 
	def _chebyshev_center(self):
		"""
		
		Returns
		-------
		center: np.ndarray(m,)
			Center of the domain
		radius: float
			radius of largest circle inside the domain
		"""
		m, n = self.A.shape

		# Merge the bound constraints into A
		A = [self.A]
		b = [self.b]
		for i in range(n):
			ei = np.zeros(n)
			ei[i] = 1
			# Upper bound
			if np.isfinite(self.ub[i]):
				A.append(ei)
				b.append(self.ub[i])
			# Lower bound
			if np.isfinite(self.lb[i]):
				A.append(-ei)
				b.append(-self.lb[i])

		A = np.vstack(A)
		b = np.hstack(b)

		if self.A_eq.shape[0] == 0:
			normA = np.sqrt( np.sum( np.power(A, 2), axis=1 ) ).reshape((A.shape[0], 1))
			AA = np.hstack(( A, normA ))
			c = np.zeros((A.shape[1]+1,))
			c[-1] = -1.0
			zc = linprog(c, A_ub = AA, b_ub = b)

			center = zc[:-1].reshape((n,))
			radius = zc[-1]
		else:
			# Rotate onto the coordinates orthogonal to the equality constraint
			Q, R = np.linalg.qr(self.A_eq.T, mode = 'complete')
			Q1 = Q[:,:self.A_eq.shape[0]]
			Q2 = Q[:,self.A_eq.shape[0]:]
			b -= np.dot(A, np.dot(self.A_eq.T, self.b_eq))
			A = np.dot(A, Q2)

			# Compute the coefficient for the center
			normA = np.sqrt( np.sum( np.power(A, 2), axis=1 ) ).reshape((A.shape[0], 1))
			AA = np.hstack(( A, normA ))
			
			c = np.zeros((A.shape[1]+1,))
			c[-1] = -1.0

			# Solve the linear program				
			zc = linprog(c, A_ub = AA, b_ub = b)
			radius = zc[-1]
			center = zc[:-1].reshape((A.shape[1],))
			
			# Convert back to the normal coorindates and add term to account for offset
			center = np.dot(Q2, center) + np.linalg.lstsq(self.A_eq, self.b_eq, rcond = -1)[0]
		self._radius = radius

		return center, radius
		
	@property
	def radius(self):
		if self._radius is None:
			self._chebyshev_center()
		return self._radius


	def _sample(self, draw = 1):
		if self._can_sample:
			X = [self._hit_and_run() for j in range(draw)]
			return np.array(X)	
		else:
			# If we cannot sample, simply return a number of points at the origin
			return np.array([self.center for j in range(draw)])	
		


	def _hit_and_run(self, recurse = True):
		""" Hit-and-run sampling for the domain
	
		self._z0 - current location of hitting and running
		"""

		# Try to find a good direction to search in
		# This direction should have a non-zero length
		# and should allow some movement through the domain

		for it in range(len(self)):
			good_dir = True

			# Choose a random direction
			p = np.random.normal(size = self._z0.shape)
			p /= np.linalg.norm(p)

			# Project the direction onto the orthogonal complement of the 
			# range of A_eq.T, so that the step continues to satisfy the 
			# equality constraint
			p -= np.dot(self._A_eq_basis, np.dot(self._A_eq_basis.T, p))
			
			# If the resulting direction is too small, try again
			if np.linalg.norm(p) < 1e-16:
				print "p too small"
				good_dir = False

			# Normalize the search direction
			p /= np.linalg.norm(p)

			if good_dir:
				# If we so far have a good step, try to find out how far along this 
				# direction we can go
				alpha_min = -self.extent(self._z0, -p)
				alpha_max = self.extent(self._z0, p)

				# If we can't go far enough in the desired direction, this is a bad direction 
				if alpha_max - alpha_min < 1e-7:
					print "extent too small", it, alpha_max - alpha_min
					good_dir = False
			
			if good_dir:
				# If our step satisfies these two requirements above, we can stop looking
				break

		# If we couldn't find a good direction, set the current location back to the center
		# and start again
		if good_dir is False and recurse:
			print "could not find good direction"
			self._z0 = np.copy(self.center)
			return self._hit_and_run(recurse = False)

		if good_dir is False and not recurse:
			return np.copy(self.center)
			#raise Exception('could not find a good direction')
		
		# Determine a new point along the direction p
		step_length = np.random.uniform(alpha_min, alpha_max)
		self._z0 += step_length * p
		if not self.isinside(self._z0):
			self._z0 = np.copy(self.center)
			if recurse:
				return self._hit_and_run(recurse = False)
			else:
				return np.copy(self._z0)
		# TODO: Ensure point _z0 still satisfies equality constraints to machine precision by solving KKT system
		return np.copy(self._z0)
		

	def _extent(self, x, p):
		return min(self._extent_bounds(x, p), self._extent_ineq(x, p))

	def _extent_bounds(self, x, p):
		""" Check the extent from the box constraints
		"""
		alpha = np.inf
		
		# To prevent divide-by-zero we ignore directions we are not moving in
		I = np.nonzero(p)

		# Check upper bounds
		y = (self.ub - x)[I]/p[I]
		if np.sum(y>0) > 0:
			alpha = min(alpha, np.min(y[y>0]))	

		# Check lower bounds
		y = (self.lb - x)[I]/p[I]
		if np.sum(y>0) > 0:
			alpha = min(alpha, np.min(y[y>0]))
		
		# If on the boundary, the direction needs to point inside the domain
		if np.any(p[self.lb == x] < 0):
			alpha = 0.
		if np.any(p[self.ub == x] > 0):
			alpha = 0.	

		return alpha	
	
	def _extent_ineq(self, x, p):
		alpha = np.inf
		# positive extent
		y = (self.b - np.dot(self.A, x)	)/np.dot(self.A, p)
		if np.sum(y>0) > 0:
			alpha = min(alpha, np.min(y[y>0]))

		return alpha


	def _isinside(self, X):
		return self._isinside_bounds(X) & self._isinside_ineq(X) & self._isinside_eq(X)

	def _isinside_bounds(self, X, tol = 1e-10):
		lb_check = np.array([np.all(x >= self.lb-tol) for x in X], dtype = np.bool)
		ub_check = np.array([np.all(x <= self.ub+tol) for x in X], dtype = np.bool)
		return lb_check & ub_check

	def _isinside_ineq(self, X, tol = 1e-10):
		return np.array([np.all(np.dot(self.A, x) <= self.b + tol) for x in X], dtype = np.bool)

	def _isinside_eq(self, X, tol = 1e-10):
		return np.array([np.all( np.abs(np.dot(self.A_eq, x) - self.b_eq) < tol) for x in X], dtype = np.bool)


	def _corner(self, p):
		x = linprog(-p, 
			A_ub = self.A,
			b_ub = self.b,
			lb = self.lb,
			ub = self.ub,
			A_eq = self.A_eq,
			b_eq = self.b_eq,
			)
		return x

	def _normalize(self, X):
		# reshape so numpy's broadcasting works correctly
		lb = self.lb.reshape(1, -1)
		ub = self.ub.reshape(1, -1)
		
		# Idenify parameters with nonzero range
		I = self.ub != self.lb
		X_norm = np.zeros(X.shape)
		# Normalize parameters with non-zero range
		X_norm[:,I] = 2.0 * (X[:,I] - lb[:,I]) / (ub[:,I] - lb[:,I]) - 1.0
		
		# The remainder should still be zero
		return X_norm

	def _unnormalize(self, X_norm, **kwargs):
		# reshape so numpy's broadcasting works correctly
		lb = self.lb.reshape(1, -1)
		ub = self.ub.reshape(1, -1)
		
		# Idenify parameters with nonzero range
		I = self.ub != self.lb
		X = np.zeros(X_norm.shape)
		# Normalize parameters with non-zero range
		X[:,I] = (ub[:,I] - lb[:,I]) * (X_norm[:,I] + 1.0)/2.0 + lb[:,I]
		X[:,~I] = lb[:,~I]
		return X 


	def _normalized_domain(self):
		# Coordiante transform
		assert np.all(np.isfinite(self.lb)) and np.all(np.isfinite(self.ub)), "Cannot normalize on an unbounded domain"
		D = np.diag( (self.ub - self.lb)/2.0)
		c = (self.lb + self.ub)/2.0

		A_norm = np.dot(self.A, D)
		b_norm = self.b - np.dot(self.A, c)
		
		A_eq_norm = np.dot(self.A_eq, D)
		b_eq_norm = self.b_eq - np.dot(self.A_eq, c)

		lb_norm = -np.ones(len(self))
		ub_norm = np.ones(len(self))
	
		return LinIneqDomain(A = A_norm, b = b_norm, lb = lb_norm, ub = ub_norm, A_eq = A_eq_norm, b_eq = b_eq_norm)


	@property
	def A(self):
		return self._A

	@property
	def b(self):
		return self._b

	@property
	def lb(self):
		return self._lb
	
	@property
	def ub(self):
		return self._ub

	@property
	def A_eq(self):
		return self._A_eq

	@property
	def b_eq(self):
		return self._b_eq
	
	def __len__(self):
		return self.lb.shape[0]


class ConvexHullDomain(LinIneqDomain):
	"""Define a domain that is the interior of a convex hull of points

	Note this is expensive in moderate dimensions (5>) due to the need to form a convex hull.
	However, by converting into a convex hull, there are fewer constraints to be satisfied.


	Parameters
	----------
	X: np.ndarray (N, m)
		Matrix of size (number of samples, number of dimensions) of data points	
	"""
	def __init__(self, X):
		self.X = np.copy(X)
		if self.X.shape[1] > 1:
			self.hull = ConvexHull(self.X) 
			A = self.hull.equations[:,:-1]
			b = -self.hull.equations[:,-1]
			LinIneqDomain.__init__(self, A, b)
			self.vertices = np.copy(self.X[self.hull.vertices])
		else:
			lb = np.atleast_1d(np.min(X))
			ub = np.atleast_1d(np.max(X))
			A = np.zeros((0, 1))
			b = np.zeros((0,))
			LinIneqDomain.__init__(self, A, b, lb = lb, ub = ub)
			self.vertices = np.array([lb, ub]).reshape(-1,1)

class ComboDomain(Domain):
	""" Holds multiple domains together
	"""
	def __init__(self, domains = None):
		if domains == None:
			domains = []
		self.domains = deepcopy(domains)
	
	def _sample(self, draw = 1):
		X = []
		for dom in self.domains:
			X.append(dom.sample(draw = draw))
		return np.hstack(X)

	def _split(self, X):
		"""Utility function for dividing coordinates X among constituent domains 

		Parameters
		----------
		X: np.ndarray(M,m)
			coordinates
		"""
		X_split = []
		start, stop = 0, 0
		for dom in self.domains:
			stop += len(dom)
			X_split.append(X.T[start:stop].T)
			start = stop

		return X_split


	def _isinside(self, X):
		inside = np.ones(X.shape[0], dtype = np.bool)
		for dom, Xdom in zip(self.domains, self._split(X)):
			inside = inside & dom.isinside(Xdom)
		return inside

	def _extent(self, x, p):
		alpha = [dom.extent(xdom, pdom) for dom, x, pdom in zip(self.domains, self._split(x), self._split(p))]
		return min(alpha)

	def _normalize(self, X):
		return np.hstack([dom.normalize(Xdom) for dom, Xdom in zip(self.domains, self._split(X))])

	def _unnormalize(self, X_norm):
		return np.hstack([dom.unnormalize(Xdom_norm) for dom, Xdom_norm in zip(self.domains, self._split(X_norm))])

	def _normalized_domain(self):
		domains_norm = [dom.normalized_domain() for dom in self.domains]
		return ComboDomain(domains = domains_norm)

	def __len__(self):
		return sum([len(dom) for dom in self.domains])
	
	def __add__(self, other):
		ret = deepcopy(self)
		assert isinstance(other, Domain)
		if isinstance(other, ComboDomain):
			ret.domains.extend(deepcopy(other.domains))
		else:
			ret.domains.append(deepcopy(other))
		return ret	

	def __radd__(self, other):
		assert isinstance(other, Domain)
		if isinstance(other, ComboDomain):
			ret = deepcopy(other)
			ret.domains.extend(deepcopy(self.domains))
		else:
			ret = deepcopy(self)
			ret.domains.insert(0,deepcopy(other))
		return ret

	@property
	def lb(self):
		return np.concatenate([dom.lb for dom in self.domains])

	@property
	def ub(self):
		return np.concatenate([dom.ub for dom in self.domains])

	@property
	def A(self):
		A = []
		for k, dom in enumerate(self.domains):
			before = int(np.sum([len(self.domains[j]) for j in range(0,k)]))
			after  = int(np.sum([len(self.domains[j]) for j in range(k+1,len(self.domains))]))
			A.append(np.hstack([np.zeros((dom.A.shape[0], before)), 
					dom.A, np.zeros((dom.A.shape[0], after))]))
		return np.vstack(A)

	@property
	def b(self):
		return np.concatenate([dom.b for dom in self.domains])

	@property
	def A_eq(self):
		A_eq = []
		for k, dom in enumerate(self.domains):
			before = int(np.sum([len(self.domains[j]) for j in range(0,k)]))
			after  = int(np.sum([len(self.domains[j]) for j in range(k+1,len(self.domains))]))
			A_eq.append(np.hstack([np.zeros((dom.A_eq.shape[0], before)), 
					dom.A_eq, np.zeros((dom.A_eq.shape[0], after))]))
		return np.vstack(A_eq)
	
	@property
	def b_eq(self):
		return np.concatenate([dom.b_eq for dom in self.domains])

	@property
	def center(self):
		return np.hstack([dom.center.flatten() for dom in self.domains])


	def _corner(self, p):
		# Check if this is a lin-ineq domain
		if all([isinstance(dom, LinIneqDomain) for dom in self.domains]):
			x = linprog(-p, 
				A_ub = self.A,
				b_ub = self.b,
				lb = self.lb,
				ub = self.ub,
				A_eq = self.A_eq,
				b_eq = self.b_eq,
				)
			return x
		else:
			raise NotImplementedError("Corners only work for linear inequality domains")


	def add_constraint(self, **kwargs):
		if all([isinstance(dom, LinIneqDomain) for dom in self.domains]):
			dom = LinIneqDomain(A = self.A, b = self.b, lb = self.lb, ub = self.ub, A_eq = self.A_eq, b_eq = self.b_eq, center = self.center)
			return dom.add_constraint(**kwargs)
		else:
			raise NotImplementedError("We currently do not support adding constraints to ComboDomains that are not linear inequality domains")


class BoxDomain(LinIneqDomain):
	def __init__(self, lb, ub, center = None):
		lb = np.array(lb).reshape(-1)
		ub = np.array(ub).reshape(-1)
		assert lb.shape[0] == ub.shape[0], "lower and upper bounds must have the same length"
		assert np.all(lb < ub)
		
		self._lb = np.copy(lb)
		self._ub = np.copy(ub)

		if center is None:
			center = (ub + lb)/2.

		self.center = np.copy(center)
		self._radius = None

	def _sample(self, draw = 1):
		x_sample = np.random.uniform(self.lb, self.ub, size = (draw, len(self)))
		return x_sample

	def _extent(self, x, p):
		return self._extent_ineq(x, p)

	def _isinside(self, X):
		return self._isinside_bounds(X)

	def _normalized_domain(self):
		return BoxDomain(-np.ones(len(self)), np.ones(len(self)))

	@property
	def A(self):
		return np.zeros((0,self.__len__()))

	@property
	def b(self):
		return np.zeros((0,))

	@property
	def A_eq(self):
		return np.zeros((0,self.__len__()))

	@property
	def b_eq(self):
		return np.zeros((0,))


class UniformDomain(BoxDomain):
	""" An alias for a box domain
	"""
	pass


class NormalDomain(BoxDomain):
	""" Domain described by a normal distribution

	Parameters
	----------
	mean : float or np.ndarray
		Mean 
	cov : float or np.ndarray
		Covariance matrix
	clip: float or None
		Truncate distribution of the standard normal to within +/-clip,
		to remove the tails.
	normalization: one of ['linear', 'nonlinear'] 
		Use either a nonlinear normalization, mapping back to the standard normal,
		or a linear mapping (which requires clip to be enabled).
	"""
	def __init__(self, mean, cov = None, clip = None, normalization = None):
		# mean
		if isinstance(mean, float) or isinstance(mean, int):
			mean = [mean]
		self.mean = np.array(mean)
		m = self.mean.shape[0]
		
		# covariance
		if isinstance(cov, float) or isinstance(cov, int):
			cov = [[cov]]
		if cov is None:
			cov = np.eye(m)
		self.cov = np.array(cov)
		assert self.cov.shape[0] == self.cov.shape[1], "Covariance must be square"
		assert self.cov.shape[0] == self.mean.shape[0], "Covariance must be the same shape as mean"

		# Check that cov is symmetric positive definite
		assert np.linalg.norm(self.cov - self.cov.T) < 1e-10
		self.ew, self.ev = np.linalg.eigh(cov)
		assert np.all(self.ew > 0), 'covariance matrix must be positive definite'
		
		self.clip = clip
		if normalization is None:
			if clip is None:
				normalization = 'nonlinear'
			else:
				normalization = 'linear'

		assert normalization in ['linear', 'nonlinear'], "normalization must be one of either 'linear' or 'nonlinear'"
		self.normalization = normalization

		if normalization == 'linear':
			assert clip is not None, "clip must be specified to use linear normalization function"
		else:
			self._normalize = self._normalize_nonlinear
			self._unnormalize = self._unnormalize_nonlinear
			self._normalized_domain = self._normalized_domain_nonlinear


	def _sample(self, draw = 1):
		X = np.random.randn(draw, self.mean.shape[0])
		if self.clip is not None:
			while True:
				# Find points that violate the clipping
				I = np.sqrt(np.sum(X**2, axis = 1)) > self.clip
				if np.sum(I) == 0:
					break
				X[I,:] = np.random.randn(np.sum(I), self.mean.shape[0])
		
		# Convert from standard normal into this domain
		X = (self.mean.reshape(-1,1) + np.dot(np.diag(np.sqrt(self.ew)), np.dot(self.ev.T, X.T))).T
		return X	

	def _normalize_nonlinear(self, X):
		return np.dot(self.ev, np.dot(np.diag(1./np.sqrt(self.ew)), X.T - self.mean.reshape(-1,1))).T

	def _unnormalize_nonlinear(self, X_norm):
		return (self.mean.reshape(-1,1) + np.dot(np.diag(np.sqrt(self.ew)), np.dot(self.ev.T, X_norm.T))).T
	
	#def _normalized_domain(self):
	#	# Linear domain transform
	#	assert self.clip is not None, "to generate normalized domain with a linear transform, clip must not be none"
	#	mean_norm = np.zeros(len(self))
	#	D = np.diag(2./(self.ub - self.lb))
	#	cov_norm = np.dot(D, np.dot(self.cov, D))
	#	return NormalDomain(mean_norm, cov_norm, clip = self.clip, normalization = self.normalization) 
	
	def _normalized_domain(self):
		assert self.normalization == 'linear'
		return BoxDomain(-1*np.ones(self.lb.shape),np.ones(self.ub.shape))

	def _normalized_domain_nonlinear(self):
		return NormalDomain(np.zeros(len(self)), np.eye(len(self)), clip = self.clip, normalization = self.normalization)

	@property		
	def lb(self):
		if self.clip is None:
			return -np.inf*np.ones(len(self))
		elif len(self) == 1:
			return self.mean.reshape(-1) - self.clip*np.sqrt(np.diag(self.cov))
		else:
			# Should loop over corner for each coordinate, and cache those values
			# this should probably be done in the initialization
			raise NotImplementedError	
	
	@property		
	def ub(self):
		if self.clip is None:
			return np.inf*np.ones(len(self))
		elif len(self) == 1:
			return self.mean.reshape(-1) + self.clip*np.sqrt(np.diag(self.cov))
		else:
			raise NotImplementedError

	@property
	def center(self):
		return self.mean.reshape(-1)

	def __len__(self):
		return self.mean.shape[0]
 
	def _isinside(self, X):
		print "called isinside"
		if self.clip is None:
			return np.ones(X.shape[0], dtype = np.bool)
		else:
			X_norm = self._normalize_nonlinear(X)
			X2 = np.sqrt(np.sum(X_norm**2, axis = 1))
			return X2 < self.clip	

	def _extent(self, x, p):
		if self.clip is None:
			return np.inf
		elif len(self) == 1:
			# If there is only one coordinate, we can simply check against the bounds
			return self._extent_bounds(x, p)
		else:
			print "called extent"
			dist_from_boundary = lambda alpha: np.linalg.norm(self._normalize_nonlinear(x + alpha*p), 2) - self.clip
			alpha = auto_root(dist_from_boundary) 	
			return alpha

	def _corner(self, p):
		if self.clip is None:
			raise UnboundedError("Cannot find the corner on an unbounded domain")
		elif len(self) == 1:
			# If there is only one coordinate, we can simply use an LP
			return linprog(-p, lb = self.lb, ub = self.ub)
		else:
			# TODO: implement quadratic program for finding the boundary
			# for a comparison of quadratic program solvers, see: https://scaron.info/blog/quadratic-programming-in-python.html
			# the conclusion is quadprog is probably the best bet: https://github.com/rmcgibbo/quadprog
			raise NotImplementedError

# TODO: Ensure sampling is still correct (IMPORTANT FOR DUU Solution)
class LogNormalDomain(NormalDomain):
	""" A domain imbued with a log normal sampling measure

	A log-normal domain is distributed like:
		
		offset + scaling * exp(x)    where x samples a normal distribution with mean mean and covariance cov

	Parameters
	----------
	mean: np.ndarray or float
		mean of normal distribution feeding the log-normal distribution
	cov: np.ndarray or float
		covariance matrix of normal distribution that feeds the log-normal distribution
	clip: float
		truncate the normal distribution feeding the log-normal distribution
	offset: np.ndarray [optional]
		Shift the distribution
	scaling: float or np.ndarray
		Scale the output of the log-normal distribution
	normalization: one of ['linear', 'nonlinear']
		Use either a nonlinear normalization, mapping back to the standard normal,
		or a linear mapping (which requires clip to be enabled).
	"""	
	def __init__(self, mean, cov, scaling = 1., offset = None, clip = None, normalization = None):
		# Since the setup is almost identical to a normal domain, we simply borrow the constructor
		NormalDomain.__init__(self, mean, cov, clip = clip, normalization = normalization)
		self.scaling = np.array(scaling).reshape(-1,1)
		if offset is None:
			offset = np.zeros(self.mean.shape)
		self.offset = offset

	def _sample(self, draw = 1):
		return self.offset + self.scaling * np.exp(NormalDomain._sample(self, draw = draw))

	@property
	def lb(self):
		if self.clip is None:
			return self.offset.reshape(-1)
		else:
			return (self.offset + self.scaling.reshape(-1) * np.exp(NormalDomain.lb.fget(self) )).reshape(-1)

	@property
	def ub(self):
		if self.clip is None:
			return np.inf*np.ones( (len(self),))
		else:
			return (self.offset + self.scaling.reshape(-1) * np.exp(NormalDomain.ub.fget(self))).reshape(-1)
	
	def _normalize_nonlinear(self, X):
		if np.any(X/self.scaling <= 0):
			print "invalid value", X/self.scaling
		return NormalDomain._normalize_nonlinear(self, np.log(X/self.scaling))

	def _unnormalize_nonlinear(self, X_norm):
		return self.scaling * np.exp(NormalDomain._unnormalize_nonlinear(self, X_norm))

# 	TODO: How can we make this code work such that we preserve density after normalization
#	def _normalized_domain(self):
#		assert self.clip is not None, "to generate normalized domain with a linear transform, clip must not be none"
#		lb = NormalDomain.lb.fget(self)
#		ub = NormalDomain.ub.fget(self)
#
#		scaling_norm = ( 2.0/(np.exp(ub) - np.exp(lb) ))
#		scaling_norm = scaling_norm.reshape(-1,1)
#		offset_norm = -np.ones(len(self)) - np.exp(lb)*scaling_norm
#
#		return LogNormalDomain(self.mean, self.cov, offset = offset_norm, scaling = scaling_norm, normalization = self.normalization, clip = self.clip)

	def _normalized_domain_nonlinear(self):
		return NormalDomain(np.zeros(len(self)), np.eye(len(self)), clip = self.clip, normalization = self.normalization)
	
	def _extent(self, x, p):
		if self.clip is None:
			return self._extent_bounds(x, p)
		else:
			NormalDomain._extent(self, x, p)
			
	def _isinside(self, X):
		print "called LogNormalDomain isinside"
		if self.clip is None:
			return np.min(X>0, axis = 1)
		else:
			X_norm = self._normalize_nonlinear(X)
			X2 = np.sqrt(np.sum(X_norm**2, axis = 1))
			return X2 < self.clip	

	@property
	def center(self):
		return self.scaling * np.exp(self.mean.reshape(-1))


