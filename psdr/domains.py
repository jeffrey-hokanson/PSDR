"""Base domain types"""
from __future__ import print_function, division

import types
import itertools
import numpy as np
from scipy.optimize import newton, brentq
from copy import deepcopy

import scipy.linalg
import scipy.stats
from scipy.optimize import nnls, minimize
from scipy.linalg import orth, solve_triangular
from scipy.spatial import ConvexHull

import cvxpy as cp
import warnings



from .quadrature import *

__all__ = ['Domain',
		'UnboundedDomain',
		'LinQuadDomain',
		'LinIneqDomain',
		'ConvexHullDomain',
		'BoxDomain', 
		'PointDomain',
		'UniformDomain',
		'RandomDomain',
		'NormalDomain',
		'LogNormalDomain',
		'TensorProductDomain',
	] 

class EmptyDomain(Exception):
	pass 

class SolverError(Exception):
	pass

def closest_point(dom, x0, L, **kwargs):
	r""" Solve the closest point problem given a domain
	"""
	x_norm = cp.Variable(len(dom))
	constraints = dom._build_constraints_norm(x_norm)
	x0_norm =  dom.normalize(x0)
	
	if L is None:
		L = np.eye(len(dom))
		
	D = dom._unnormalize_der() 	
	LD = L.dot(D)
	obj = cp.norm(LD*x_norm - LD.dot(x0_norm))

	# There is a bug in cvxpy causing these deprecation warnings to appear
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', PendingDeprecationWarning)
		problem = cp.Problem(cp.Minimize(obj), constraints)
		problem.solve(**kwargs)

	# TODO: Check solution state 			
	return dom.unnormalize(np.array(x_norm.value).reshape(len(dom)))	

def constrained_least_squares(dom, A, b, **kwargs):
	x_norm = cp.Variable(len(dom))
	D = dom._unnormalize_der() 
	c = dom._center()	
	# There is a bug in cvxpy causing these deprecation warnings to appear
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', PendingDeprecationWarning)
		# \| A x - b\|_2 
		obj = cp.norm(x_norm.__rmatmul__(A.dot(D)) - b - A.dot(c) )
		constraints = dom._build_constraints_norm(x_norm)
		problem = cp.Problem(cp.Minimize(obj), constraints)
		problem.solve(**kwargs)
	return dom.unnormalize(np.array(x_norm.value).reshape(len(dom)))

def corner(dom, p, **kwargs):
	x_norm = cp.Variable(len(dom))
	D = dom._unnormalize_der() 	
	# There is a bug in cvxpy causing these deprecation warnings to appear
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', PendingDeprecationWarning)
		# p.T @ x
		if len(dom) > 1:
			obj = x_norm.__rmatmul__(D.dot(p).reshape(1,-1))
		else:
			obj = x_norm*float(D.dot(p))
		constraints = dom._build_constraints_norm(x_norm)
		problem = cp.Problem(cp.Maximize(obj), constraints)
		problem.solve(**kwargs)
	if problem.status != 'optimal':
		raise SolverError
	return dom.unnormalize(np.array(x_norm.value).reshape(len(dom)))




class Domain(object):
	r""" Abstract base class for an input domain

	This specifies a domain :math:`\mathcal{D}\subset \mathbb{R}^m`.

	"""

	@property
	def names(self):
		try:
			return self._names
		except AttributeError:
			self._names = ['x%d' % i for i in range(len(self))]
			return self._names

	def _init_names(self, names):
		if names is None:
			return

		if isinstance(names, str):
			assert len(self) == 1, "A list of names must be provided for domains with greater than one dimension" 
			self._names = [names]
		else:
			assert len(self) == len(names), "Number of names must match dimension"
			self._names = names
	
	def __len__(self):
		raise NotImplementedError

	# To define the documentation once for all domains, these functions call internal functions
	# to each subclass
	
	def closest_point(self, x0, L = None, **kwargs):
		r"""Given a point, find the closest point in the domain to it.

		Given a point :math:`\mathbf x_0`, find the closest point :math:`\mathbf x`
		in the domain :math:`\mathcal D` to it by solving the optimization problem

		.. math::
		
			\min_{\mathbf x \in \mathcal D} \| \mathbf L (\mathbf x - \mathbf x_0)\|_2

		where :math:`\mathbf L` is an optional weighting matrix.		

		Parameters
		----------
		x0: array-like
			Point in :math:`\mathbb R^m`  
		L: array-like, optional
			Matrix of size (p,m) to use as a weighting matrix in the 2-norm;
			if not provided, the standard 2-norm is used.
		kwargs: dict, optional
			Additional arguments to pass to the optimizer
		def _sample(self, draw = 1):
		return self.domain_norm._sample(draw)	
		Returns
		-------
		x: array-like
			Coordinates of closest point in this domain to :math:`\mathbf x_0`

		Raises
		------
		ValueError
		"""
		try: 
			x0 = np.array(x0).reshape(len(self))
		except ValueError:
			raise ValueError('Dimension of x0 does not match dimension of the domain')

		if L is not None:
			try: 
				L = np.array(L).reshape(-1,len(self))
			except ValueError:
				raise ValueError('The second dimension of L does not match that of the domain')
 
		return self._closest_point(x0, L = L, **kwargs)

	def _closest_point(self, x0, L = None, **kwargs):
		raise NotImplementedError	
	
	def corner(self, p, **kwargs):
		r""" Find the point furthest in direction p inside the domain

		Given a direction :math:`\mathbf p`, find the point furthest away in that direction

		.. math::
 	
			\max_{\mathbf{x} \in \mathcal D}  \mathbf{p}^\top \mathbf{x}

		Parameters
		----------
		p: array-like (m,)
			Direction in which to search for furthest point
		kwargs: dict, optional
			Additional parameters to be passed to cvxpy solve
		"""
		try:
			p = np.array(p).reshape(len(self))
		except ValueError:
			raise ValueError("Dimension of search direction doesn't match the domain dimension")

		return self._corner(p, **kwargs)
	
	def _corner(self, p, **kwargs):
		raise NotImplementedError	


	def extent(self, x, p):
		r"""Compute the distance alpha such that x + alpha * p is on the boundary of the domain

		Given a point :math:`\mathbf{x}\in\mathcal D` and a direction :math:`\mathbf p`,
		find the furthest we can go in direction :math:`\mathbf p` and stay inside the domain:

		.. math::
	
			\max_{\alpha > 0}   \alpha \quad\\text{such that} \quad \mathbf x +\\alpha \mathbf p \in \mathcal D

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
		try:
			x = np.array(x).reshape(len(self))
		except ValueError:
			raise ValueError("Starting point not the same dimension as the domain")

		assert self.isinside(x), "Starting point must be inside the domain" 
		return self._extent(x, p)

	def _extent(self, x, p):
		raise NotImplementedError

	def isinside(self, X):
		""" Determine if points are inside the domain

		Parameters
		----------
		X : np.ndarray(M, m)
			Samples in rows of X
		"""
		# Make this a 2-d array
		X = np.atleast_1d(X)
		if len(X.shape) == 1:
			# Check for dimension mismatch
			if X.shape[0] != len(self):
				return False
			X = X.reshape(-1, len(self)) 	
			return self._isinside(X).flatten()
		else:
			# Check if the dimensions match
			if X.shape[1] != len(self):
				return np.zeros(X.shape[0], dtype = np.bool)
			return self._isinside(X)

	def _isinside(self, X):
		raise NotImplementedError

	def normalize(self, X):
		""" Given a points in the application space, convert it to normalized units
		
		Parameters
		----------
		X: np.ndarray((M,m))
			points in the domain to normalize
		"""
		try:
			X.shape
		except AttributeError:
			X = np.array(X)
		if len(X.shape) == 1:
			X = X.reshape(-1, len(self)) 
			return self._normalize(X).flatten()
		else:
			return self._normalize(X)

	def _normalize(self, X):
		raise NotImplementedError

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
	
	def _unnormalize(self, X_norm):
		raise NotImplementedError
	
	def normalized_domain(self):
		""" Return a domain with units normalized corresponding to this domain
		"""
		return self._normalized_domain()
	
	def __mul__(self, other):
		""" Combine two domains
		"""
		return TensorProductDomain([self, other])
	
	def __rmul__(self, other):
		""" Combine two domains
		"""
		return TensorProductDomain([self, other])
	
	def constrained_least_squares(self, A, b, **kwargs):
		r"""Solves a least squares problem constrained to the domain

		Given a matrix :math:`\mathbf{A} \in \mathbb{R}^{n\times m}`
		and vector :math:`\mathbf{b} \in \mathbb{R}^n`,
		solve least squares problem where the solution :math:`\mathbf{x}\in \mathbb{R}^m`
		is constrained to the domain :math:`\mathcal{D}`:

		.. math::
		
			\min_{\mathbf{x} \in \mathcal{D}} \| \mathbf{A} \mathbf{x} - \mathbf{b}\|_2^2
		
		Parameters
		----------
		A: array-like (n,m)	
			Matrix in least squares problem
		b: array-like (n,)
			Right hand side of least squares problem
		kwargs: dict, optional
			Additional parameters to pass to solver
		""" 
		try:
			A = np.array(A).reshape(-1,len(self))
		except ValueError:
			raise ValueError("Dimension of matrix A does not match that of the domain")
		try:
			b = np.array(b).reshape(A.shape[0])
		except ValueError:
			raise ValueError("dimension of b in least squares problem doesn't match A")

		return self._constrained_least_squares(A, b, **kwargs)	

	def sample(self, draw = 1):
		""" Sample points with uniform probability from the measure associated with the domain.

		This is intended as a low-level interface for generating points from the domain.
		More advanced approaches are handled through the Sampler subclasses.

		Parameters
		----------
		draw: int
			Number of samples to return

		Returns
		-------
		array-like (draw, len(self))
			Array of samples from the domain
		"""
		x_sample = self._sample(draw = int(draw))
		if draw == 1: 
			x_sample = x_sample.flatten()
		return x_sample
	
	def _sample(self, draw = None):
		# By default, use the hit and run sampler
		if draw is None:
			draw = 1

		X = [self._hit_and_run() for i in range(3*draw)]
		I = np.random.permutation(len(X))
		return np.array([X[i] for i in I[0:draw]])


	def sample_grid(self, n):
		r""" Sample points from a tensor product grid inside the domain
	
		For a bounded domain this function provides samples that come from a uniformly spaced grid.
		This grid contains `n` points in each direction, linearly spaced between the lower and upper bounds.
		For box domains, this will contain $n^d$ samples where $d$ is the dimension of the domain.
		For other domains, this will potentially contain fewer samples since points on the grid outside the domain
		are excluded.
	
		Parameters
		----------
		n: int
			Number of samples in each direction
		"""

		assert np.all(np.isfinite(self.lb)) & np.all(np.isfinite(self.ub)), "Requires a bounded domain"
		xs = [np.linspace(lbi, ubi, n) for lbi, ubi in zip(self.lb, self.ub)]
		Xs = np.meshgrid(*xs, indexing = 'ij')
		Xgrid = np.vstack([X.flatten() for X in Xs]).T
		I = self.isinside(Xgrid)
		return Xgrid[I]	

	def quadrature_rule(self, N):
		r""" Constructs quadrature rule for the domain

		Given N, constructs a Monte-Carlo quadrature rule for $M \le N$ such that 

		.. math::
		
			\int_{\mathbf x\in \mathcal D} f(\mathbb{x}) \mathrm{d}\mathbf{x}
			\approx \sum_{j=1}^M w_j f(\mathbf{x}_j).
		
		Parameters
		----------
		N: int
			Number of samples to use to construct estimate

		Returns
		-------
		X: np.ndarray (M, len(self))
			Samples from the domain
		w: np.ndarray (M,)
			Weights for quadrature rule

		"""
		M = int(N)
		w = (1./M)*np.ones(M)
	
		X = self.sample(M)
		return X, w


	def _hit_and_run(self, _recurse = 2):
		r"""Hit-and-run sampling for the domain
		"""
		if _recurse < 0:
			raise ValueError("Could not find valid hit and run step")

		try:
			# Get the current location of where the hit and run sampler is
			x0 = self._hit_and_run_state
			if x0 is None: raise AttributeError
		except AttributeError:
			# If this hasn't been initialized find a feasible starting point
			N = 10
			# In earlier versions, we find the starting point by finding the Chebeychev center;
			# here we use a simpler approach that simply picks N points on the boundary 
			# by calling corner and then take the mean (since the domain is convex).
			# This removes the need to treat equality constraints carefully and also
			# generalizes to LinQuadDomains. 
			x0 = sum([self.corner(np.random.randn(len(self))) for i in range(N)])/N
			x0 = self.closest_point(x0)
			self._hit_and_run_state = x0
			
	
		# Sometimes we may have a point that is slightly outside due to numerical issues
		# so we push it back in
		if not self.isinside(x0):
			x0 = self.closest_point(x0) 	

		# See if there is an orthongonal basis for the equality constraints
		# This is necessary so we can generate random directions that satisfy the equality constraint.
		# TODO: Should we generalize this as a "tangent cone" or "feasible cone" that each domain implements?
		try:
			Qeq = self._A_eq_basis
		except AttributeError:
			try: 
				if len(self.A_eq) == 0: raise AttributeError
				Qeq = orth(self.A_eq.T)
			except AttributeError:
				Qeq = np.zeros((len(self),0))
			self._A_eq_basis = Qeq
			

		# Loop over multiple search directions if we have trouble 
		for it in range(len(self)):	
			p = np.random.normal(size = (len(self),))
			# Orthogonalize against equality constarints constraints
			p = p - Qeq.dot(Qeq.T.dot(p))
			p /= np.linalg.norm(p)

			alpha_min = -self.extent(x0, -p)
			alpha_max =  self.extent(x0,  p)
			
			if alpha_max - alpha_min > 1e-7:
				alpha = np.random.uniform(alpha_min, alpha_max)
				self._hit_and_run_state += alpha*p
				return np.copy(self._hit_and_run_state)	
		
		# If we've failed to find a good direction, reinitialize, and recurse
		self._hit_and_run_state = None
		return self._hit_and_run(_recurse = _recurse - 1)


	################################################################################		
	# Simple properties
	################################################################################		
	
	@property
	def lb(self): return -np.inf*np.ones(len(self)) 

	@property
	def ub(self): return np.inf*np.ones(len(self)) 

	@property
	def A(self): return np.zeros((0, len(self)))

	@property
	def b(self): return np.zeros((0,))

	@property
	def A_eq(self): return np.zeros((0, len(self)))

	@property
	def b_eq(self): return np.zeros((0,))

	@property
	def Ls(self): return ()

	@property
	def ys(self): return ()
	
	@property
	def rhos(self): return ()
	
	@property
	def lb_norm(self):
		return self.normalize(self.lb)

	@property
	def ub_norm(self):
		return self.normalize(self.ub)

	@property
	def A_norm(self):
		D = self._unnormalize_der()
		return self.A.dot(D)

	@property
	def b_norm(self):
		c = self._center()
		return self.b - self.A.dot(c)

	@property
	def A_eq_norm(self):	
		D = self._unnormalize_der()
		return self.A_eq.dot(D)

	@property
	def b_eq_norm(self):
		c = self._center()
		return self.b_eq - self.A_eq.dot(c)

	@property
	def Ls_norm(self):
		D = self._unnormalize_der()
		return [ L.dot(D) for L in self.Ls]
			
	@property
	def ys_norm(self):
		c = self._center()
		return [y - c for y in self.ys]	

	@property
	def rhos_norm(self):
		return self.rhos



	# These are the lower and upper bounds to use for normalization purposes;
	# they do not add constraints onto the domain.
	@property
	def norm_lb(self):
		r"""Lower bound used for normalization purposes; does not constrain the domain
		"""
		try: 
			return self._norm_lb
		except AttributeError:
			self._norm_lb = -np.inf*np.ones(len(self))
			for i in range(len(self)):
				ei = np.zeros(len(self))
				ei[i] = 1
				if np.isfinite(self.lb[i]):
					self._norm_lb[i] = self.lb[i]
				else:
					try:
						x_corner = self.corner(-ei)
						self._norm_lb[i] = x_corner[i]	
					except SolverError:
						self._norm_lb[i] = -np.inf

			return self._norm_lb

	@property
	def norm_ub(self):
		r"""Upper bound used for normalization purposes; does not constrain the domain
		"""
		try: 
			return self._norm_ub
		except AttributeError:
			# Temporarly disable normalization
			
			# Note: since this will be called by corner, we need to 
			# choose a reasonable value to initialize this property, which
			# will be used until the remainder of the corner calls are made
			self._norm_ub = np.inf*np.ones(len(self))
			for i in range(len(self)):
				ei = np.zeros(len(self))
				ei[i] = 1
				if np.isfinite(self.ub[i]):
					self._norm_ub[i] = self.ub[i]
				else:
					try:
						x_corner = self.corner(ei)
						self._norm_ub[i] = x_corner[i]	
					except SolverError:
						self._norm_ub[i] = np.inf
			
			return self._norm_ub
	
	################################################################################		
	# Normalization functions 
	################################################################################		

	def isnormalized(self):
		return np.all( (~np.isfinite(self.norm_lb)) | (self.norm_lb == -1.) ) and np.all( (~np.isfinite(self.norm_ub)) | (self.norm_ub == 1.) ) 

	def _normalize_der(self):
		"""Derivative of normalization function"""

		slope = np.ones(len(self))
		I = (self.norm_ub != self.norm_lb) & np.isfinite(self.norm_lb) & np.isfinite(self.norm_ub)
		slope[I] = 2.0/(self.norm_ub[I] - self.norm_lb[I])
		return np.diag(slope)

	def _unnormalize_der(self):
		slope = np.ones(len(self))
		I = (self.norm_ub != self.norm_lb) & np.isfinite(self.norm_lb) & np.isfinite(self.norm_ub)
		slope[I] = (self.norm_ub[I] - self.norm_lb[I])/2.0
		return np.diag(slope)
	
	def _center(self):
		c = np.zeros(len(self))
		I = np.isfinite(self.norm_lb) & np.isfinite(self.norm_ub)
		c[I] = (self.norm_lb[I] + self.norm_ub[I])/2.0
		return c	

	def _normalize(self, X):
		# reshape so numpy's broadcasting works correctly
		#lb = self.norm_lb.reshape(1, -1)
		#ub = self.norm_ub.reshape(1, -1)
		
		# Those points with zero range get mapped to zero, so we only work on those
		# with a non-zero range
		#X_norm = np.zeros(X.shape)
		#I = (self.norm_ub != self.norm_lb) & np.isfinite(self.norm_lb) & np.isfinite(self.norm_ub)
		#X_norm[:,I] = 2.0 * (X[:,I] - lb[:,I]) / (ub[:,I] - lb[:,I]) - 1.0
		#I = (self.norm_ub != self.norm_lb)
		#X_norm[:,I] = 0
		#I = ~np.isfinite(self.norm_lb) | ~np.isfinite(self.norm_ub)
		#X_norm[:,I] = X[:,I]
		
		c = self._center()
		D = self._normalize_der()
		X_norm = D.dot( (X - c.reshape(1,-1)).T ).T
		return X_norm
	
	def _unnormalize(self, X_norm, **kwargs):
#		# reshape so numpy's broadcasting works correctly
#		lb = self.norm_lb.reshape(1, -1)
#		ub = self.norm_ub.reshape(1, -1)
#		
#		# Idenify parameters with nonzero range
#		X = np.zeros(X_norm.shape)
#
#		# unnormalize parameters with non-zero range and bounded
#		I = (self.norm_ub != self.norm_lb) & np.isfinite(self.norm_lb) & np.isfinite(self.norm_ub)
#		X[:,I] = (ub[:,I] - lb[:,I]) * (X_norm[:,I] + 1.0)/2.0 + lb[:,I]
#	
#		# for those with infinite bounds, apply no transformation
#		I = ~np.isfinite(self.norm_lb) | ~np.isfinite(self.norm_ub) 
#		X[:,I] = X_norm[:,I]	
#		# for those dimensions with zero dimension, set to the center point lb[:,~I] = ub[:,~I]
#		I = (self.norm_ub == self.norm_lb)
#		X[:,I] = lb[:,I]
		
		c = self._center()
		Dinv = self._unnormalize_der()
		X = Dinv.dot(X_norm.T).T + c.reshape(1,-1)
		return X 

	################################################################################		
	# Bound checking
	################################################################################		
	def _isinside_bounds(self, X, tol = None):
		if tol is None: tol = self.tol
		lb_check = np.array([np.all(x >= self.lb-tol) for x in X], dtype = np.bool)
		ub_check = np.array([np.all(x <= self.ub+tol) for x in X], dtype = np.bool)
		return lb_check & ub_check

	def _isinside_ineq(self, X, tol = None):
		if tol is None: tol = self.tol
		return np.array([np.all(np.dot(self.A, x) <= self.b + tol) for x in X], dtype = np.bool)

	def _isinside_eq(self, X, tol = None):
		if tol is None: tol = self.tol
		return np.array([np.all( np.abs(np.dot(self.A_eq, x) - self.b_eq) < tol) for x in X], dtype = np.bool)

	def _isinside_quad(self, X, tol = None):
		"""check that points are inside quadratic constraints"""
		if tol is None: tol = self.tol
		inside = np.ones(X.shape[0],dtype = np.bool)
		for L, y, rho in zip(self.Ls, self.ys, self.rhos):
			diff = X - np.tile(y.reshape(1,-1), (X.shape[0],1))
			Ldiff = L.dot(diff.T).T
			Ldiff_norm = np.sum(Ldiff**2,axis=1)
			inside = inside & (np.sqrt(Ldiff_norm) <= rho + tol)
		return inside
	

	################################################################################		
	# Extent functions 
	################################################################################		
	
	def _extent_bounds(self, x, p):
		"""Check the extent from the box constraints"""
		alpha = np.inf
		
		# If on the boundary, the direction needs to point inside the domain
		# otherwise we cannot move
		if np.any(p[self.lb == x] < 0):
			return 0.
		if np.any(p[self.ub == x] > 0):
			return 0.	
		
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
		
		return alpha

	def _extent_ineq(self, x, p):
		""" check the extent from the inequality constraints """
		alpha = np.inf
		# positive extent
		y = (self.b - np.dot(self.A, x)	)/np.dot(self.A, p)
		if np.sum(y>0) > 0:
			alpha = min(alpha, np.min(y[y>0]))

		return alpha
	
	def _extent_quad(self, x, p):
		""" check the extent from the quadratic constraints"""
		alpha = np.inf
		for L, y, rho in zip(self.Ls, self.ys, self.rhos):
			Lp = L.dot(p)
			Lxy = L.dot(x - y)
			# Terms in quadratic formula a alpha^2 + b alpha + c
			a = Lp.T.dot(Lp)
			b = 2*Lp.T.dot(Lxy)
			c = Lxy.T.dot(Lxy) - rho**2
			
			roots = np.roots([a,b,c])
			real_roots = roots[np.isreal(roots)]
			pos_roots = real_roots[real_roots>=0]
			if len(pos_roots) > 0:
				alpha = min(alpha, min(pos_roots))
		return alpha

class UnboundedDomain(Domain):
	r""" A domain without any constraints
	
	This class implements a subset of the functionality of the Domain
	applicable for a domain that is all of :math:`\mathbb{R}^m`.

	Parameters
	----------


	"""
	def __init__(self, dimension, names = None):
		self._dimension = dimension
		self._init_names(names)
	
	def __len__(self):
		return self._dimension

	def _build_constraints(self, x):
		return [] 
	
	def _build_constraints_norm(self, x_norm):
		return []

	def _closest_point(self, x0, L = None, **kwargs):
		return x0

	def _normalize(self, X):
		return X

	def _unnormalize(self, X_norm):
		return X_norm

	def _isinside(self, X):
		if X.shape[1]== len(self):
			return np.ones(X.shape[0],dtype = np.bool)
		else:
			return np.zeros(X.shape[0],dtype = np.bool)

class LinQuadDomain(Domain):
	r"""A domain specified by a combination of linear (in)equality constraints and convex quadratic constraints


	Here we define a domain that is specified in terms of bound constraints,
	linear inequality constraints, linear equality constraints, and quadratic constraints.

	.. math::

		\mathcal{D} := \left \lbrace
			\mathbf{x} : \text{lb} \le \mathbf{x} \le \text{ub}, \ 
			\mathbf{A} \mathbf{x} \le \mathbf{b}, \
			\mathbf{A}_{\text{eq}} \mathbf{x} = \mathbf{b}_{\text{eq}}, \
			\| \mathbf{L}_i (\mathbf{x} - \mathbf{y}_i)\|_2 \le \rho_i
		\right\rbrace \subset \mathbb{R}^m


	Parameters
	----------
	A: array-like (m,n)
		Matrix in left-hand side of inequality constraint
	b: array-like (m,)
		Vector in right-hand side of the ineqaluty constraint
	A_eq: array-like (p,n)
		Matrix in left-hand side of equality constraint
	b_eq: array-like (p,) 
		Vector in right-hand side of equality constraint
	lb: array-like (n,)
		Vector of lower bounds 
	ub: array-like (n,)
		Vector of upper bounds 
	Ls: list of array-likes (p,m)
		List of matrices with m columns defining the quadratic constraints
	ys: list of array-likes (m,)
		Centers of the quadratic constraints
	rhos: list of positive floats 
		Radii of quadratic constraints
	kwargs: dict, optional
		Additional parameters to be passed to cvxpy Problem.solve() 
	"""
	def __init__(self, A = None, b = None, 
		lb = None, ub = None, 
		A_eq = None, b_eq = None, 
		Ls = None, ys = None, rhos = None,
		names = None, **kwargs):

		self.tol = 1e-6
		# Determine dimension of space
		self._init_dim(lb = lb, ub = ub, A = A, A_eq = A_eq, Ls = Ls)

		# Start setting default values
		self._lb = self._init_lb(lb)
		self._ub = self._init_ub(ub)
		self._A, self._b = self._init_ineq(A, b)
		self._A_eq, self._b_eq = self._init_eq(A_eq, b_eq)	
		self._Ls, self._ys, self._rhos = self._init_quad(Ls, ys, rhos)
		
		self._init_names(names)	

		if len(kwargs) == 0:
			#kwargs= {'solver': cp.CVXOPT, 'reltol': 1e-10, 'abstol' : 1e-10, 'verbose': False}
			kwargs ={} 
		self.kwargs = kwargs
	
	
	################################################################################		
	# Initialization helpers 
	################################################################################		
	def _init_dim(self, lb = None, ub = None, A = None, A_eq = None, Ls = None):
		"""determine the dimension of the space we are working on"""
		if lb is not None:
			if isinstance(lb, (int, float)):
				m = 1
			else:
				m = len(lb)
		elif ub is not None:
			if isinstance(ub, (int, float)):
				m = 1
			else:
				m = len(ub)
		elif A is not None:
			m = len(A[0])
		elif A_eq is not None:
			m = len(A_eq[0])
		elif Ls is not None:
			m = len(Ls[0][0])
		else:
			raise Exception("Could not determine dimension of space")

		self._dimension = m


	def _init_lb(self, lb):
		if lb is None:
			return -np.inf*np.ones(len(self))
		else:
			if isinstance(lb, (int, float)):
				lb = [lb]
			assert len(lb) == len(self), "Lower bound has wrong dimensions"
			return np.array(lb)
		
	def _init_ub(self, ub):
		if ub is None:
			return np.inf*np.ones(len(self))
		else:
			if isinstance(ub, (int, float)):
				ub = [ub]
			assert len(ub) == len(self), "Upper bound has wrong dimensions"
			return np.array(ub)
		
	def _init_ineq(self, A, b):
		if A is None and b is None:
			A = np.zeros((0,len(self)))
			b = np.zeros((0,))
		elif A is not None and b is not None:
			A = np.array(A)
			b = np.array(b)
			if len(b.shape) == 0:
				b = b.reshape(1)

			assert len(b.shape) == 1, "b must have only one dimension"
			
			if len(A.shape) == 1 and len(b) == 1:
				A = A.reshape(1,-1)
	
			assert A.shape[1] == len(self), "A has wrong number of columns"
			assert A.shape[0] == b.shape[0], "The number of rows of A and b do not match"
		else:
			raise AssertionError("If using inequality constraints, both A and b must be specified")
		return A, b	
	
	def _init_eq(self, A_eq, b_eq):
		if A_eq is None and b_eq is None:
			A_eq = np.zeros((0,len(self)))
			b_eq = np.zeros((0,))
		elif A_eq is not None and b_eq is not None:
			A_eq = np.array(A_eq)
			b_eq = np.array(b_eq)
			if len(b_eq.shape) == 0:
				b_eq = b_eq.reshape(1)

			assert len(b_eq.shape) == 1, "b_eq must have only one dimension"
			
			if len(A_eq.shape) == 1 and len(b_eq) == 1:
				A_eq = A_eq.reshape(1,-1)

			assert A_eq.shape[1] == len(self), "A_eq has wrong number of columns"
			assert A_eq.shape[0] == b_eq.shape[0], "The number of rows of A_eq and b_eq do not match"
		else:
			raise AssertionError("If using equality constraints, both A_eq and b_eq must be specified")
		
		return A_eq, b_eq

	def _init_quad(self, Ls, ys, rhos):
		if Ls is None and ys is None and rhos is None:
			_Ls = []
			_ys = []
			_rhos = []
		elif Ls is not None and ys is not None and rhos is not None:
			assert len(Ls) == len(ys) == len(rhos), "Length of all quadratic constraints must be the same"
			
			_Ls = []
			_ys = []
			_rhos = []
			for L, y, rho in zip(Ls, ys, rhos):
				assert len(L[0]) == len(self), "dimension of L doesn't match the domain"
				assert len(y) == len(self), "Dimension of center doesn't match the domain"
				assert rho > 0, "Radius must be positive"
				_Ls.append(np.array(L))
				_ys.append(np.array(y))
				_rhos.append(rho)
				# TODO: If constraint is rank-1, should we implicitly convert to a linear inequality constriant
		else:
			raise AssertionError("If providing quadratic constraint, each of Ls, ys, and rhos must be defined") 
		return _Ls, _ys, _rhos 

	################################################################################		
	# Simple properties
	################################################################################		
	def __len__(self): return self._dimension
	
	@property
	def lb(self): return self._lb
	
	@property
	def ub(self): return self._ub

	@property
	def A(self): return self._A

	@property
	def b(self): return self._b

	@property
	def A_eq(self): return self._A_eq

	@property
	def b_eq(self): return self._b_eq

	@property
	def Ls(self): return self._Ls

	@property
	def ys(self): return self._ys
	
	@property
	def rhos(self): return self._rhos

		
	################################################################################		
	# Normalization 
	################################################################################		
	def _normalized_domain(self):
		names_norm = [name + ' (normalized)' for name in self.names]
		return LinQuadDomain(lb = self.lb_norm, ub = self.ub_norm, A = self.A_norm, b = self.b_norm, 
			A_eq = self.A_eq_norm, b_eq = self.b_eq_norm, Ls = self.Ls_norm, ys = self.ys_norm, rhos = self.rhos_norm,
			names = names_norm)

	def _isinside(self, X):
		return self._isinside_bounds(X) & self._isinside_ineq(X) & self._isinside_eq(X) & self._isinside_quad(X)

	def _extent(self, x, p):
		# Check that direction satisfies equality constraints to a tolerance
		if self.A_eq.shape[0] == 0 or np.all(np.abs(self.A_eq.dot(p) ) < self.tol):
			return min(self._extent_bounds(x, p), self._extent_ineq(x, p), self._extent_quad(x, p))
		else:
			return 0. 

	################################################################################		
	# Convex Solver Functions 
	################################################################################		

	def _build_constraints_norm(self, x_norm):
		r""" Build the constraints corresponding to the domain given a vector x
		"""
		constraints = []
		
		# Numerical issues emerge with unbounded constraints
		I = np.isfinite(self.lb_norm)
		if np.sum(I) > 0:
			constraints.append( self.lb_norm[I] <= x_norm[I])
		
		I = np.isfinite(self.ub_norm)
		if np.sum(I) > 0:
			constraints.append( x_norm[I] <= self.ub_norm[I])
		
		if self.A.shape[0] > 0:	
			constraints.append( x_norm.__rmatmul__(self.A_norm) <= self.b_norm)
		if self.A_eq.shape[0] > 0:
			constraints.append( x_norm.__rmatmul__(self.A_eq) == self.b_eq)

		for L, y, rho in zip(self.Ls_norm, self.ys_norm, self.rhos_norm):
			if len(L) > 1:
				constraints.append( cp.norm(x_norm.__rmatmul__(L) - L.dot(y)) <= rho )
			elif len(L) == 1:
				constraints.append( cp.norm(L*x_norm - L.dot(y)) <= rho)

		return constraints
	

	def _build_constraints(self, x):
		r""" Build the constraints corresponding to the domain given a vector x
		"""
		constraints = []
		
		# Numerical issues emerge with unbounded constraints
		I = np.isfinite(self.lb)
		if np.sum(I) > 0:
			constraints.append( self.lb[I] <= x[I])
		
		I = np.isfinite(self.ub)
		if np.sum(I) > 0:
			constraints.append( x[I] <= self.ub[I])
		
		if self.A.shape[0] > 0:	
			constraints.append( x.__rmatmul__(self.A) <= self.b)
		if self.A_eq.shape[0] > 0:
			constraints.append( x.__rmatmul__(self.A_eq) == self.b_eq)

		for L, y, rho in zip(self.Ls, self.ys, self.rhos):
			if len(L) > 1:
				constraints.append( cp.norm(x.__rmatmul__(L) - L.dot(y)) <= rho )
			elif len(L) == 1:
				constraints.append( cp.norm(L*x - L.dot(y)) <= rho)

		return constraints
	

	def _closest_point(self, x0, L = None, **kwargs):
		if len(kwargs) == 0:
			kwargs = self.kwargs
		return closest_point(self, x0, L = L, **kwargs)


	def _corner(self, p, **kwargs):
		if len(kwargs) == 0:
			kwargs = self.kwargs
 
		return corner(self, p, **kwargs)

	def _constrained_least_squares(self, A, b, **kwargs):
		if len(kwargs) == 0:
			kwargs = self.kwargs

		return constrained_least_squares(self, A, b, **kwargs)

	################################################################################		
	# 
	################################################################################		

	def add_constraints(self, A = None, b = None, lb = None, ub = None, A_eq = None, b_eq = None,
		Ls = None, ys = None, rhos = None):
		r"""Add new constraints to the domain
		"""
		lb = self._init_lb(lb)
		ub = self._init_ub(ub)
		A, b = self._init_ineq(A, b)
		A_eq, b_eq = self._init_eq(A_eq, b_eq)
		Ls, ys, rhos = self._init_quad(Ls, ys, rhos)

		# Update constraints
		lb = np.maximum(lb, self.lb)
		ub = np.minimum(ub, self.ub)

		A = np.vstack([self.A, A])	
		b = np.hstack([self.b, b])

		A_eq = np.vstack([self.A_eq, A_eq])
		b_eq = np.hstack([self.b_eq, b_eq])

		Ls = self.Ls + Ls
		ys = self.ys + ys
		rhos = self.rhos + rhos

		if len(Ls) > 0:
			return LinQuadDomain(lb = lb, ub = ub, A = A, b = b, A_eq = A_eq, b_eq = b_eq,
				 Ls = Ls, ys = ys, rhos = rhos)
		elif len(b) > 0 or len(b_eq) > 0:
			return LinIneqDomain(lb = lb, ub = ub, A = A, b = b, A_eq = A_eq, b_eq = b_eq)
		else:
			return BoxDomain(lb = lb, ub = ub)

	def __and__(self, other):
		if isinstance(other, LinQuadDomain) or (isinstance(other, TensorProductDomain) and other._is_linquad()):
			return self.add_constraints(lb = other.lb, ub = other.ub,
				A = other.A, b = other.b, A_eq = other.A_eq, b_eq = other.b_eq,
				Ls = other.Ls, ys = other.ys, rhos = other.rhos)
		else:
			raise NotImplementedError

	def __rand__(self, other):
		return self.__and__(self, other)
		
	################################################################################		
	# End of LinQuadDomain 
	################################################################################		


class LinIneqDomain(LinQuadDomain):
	r"""A domain specified by a combination of linear equality and inequality constraints.

	Here we build a domain specified by three kinds of constraints:
	bound constraints :math:`\text{lb} \le \mathbf{x} \le \text{ub}`,
	inequality constraints :math:`\mathbf{A} \mathbf{x} \le \mathbf{b}`,
	and equality constraints :math:`\mathbf{A}_{\text{eq}} \mathbf{x} = \mathbf{b}_{\text{eq}}`:
	
	.. math::

		\mathcal{D} := \left \lbrace
			\mathbf{x} : \text{lb} \le \mathbf{x} \le \text{ub}, \ 
			\mathbf{A} \mathbf{x} \le \mathbf{b}, \
			\mathbf{A}_{\text{eq}} \mathbf{x} = \mathbf{b}_{\text{eq}}
		\right\rbrace \subset \mathbb{R}^m

	Parameters
	----------
	A: array-like (m,n)
		Matrix in left-hand side of inequality constraint
	b: array-like (m,)
		Vector in right-hand side of the ineqaluty constraint
	A_eq: array-like (p,n)
		Matrix in left-hand side of equality constraint
	b_eq: array-like (p,) 
		Vector in right-hand side of equality constraint
	lb: array-like (n,)
		Vector of lower bounds 
	ub: array-like (n,)
		Vector of upper bounds 
	kwargs: dict, optional
		Additional parameters to pass to solvers 

	"""
	def __init__(self, A = None, b = None, lb = None, ub = None, A_eq = None, b_eq = None, names = None, **kwargs):
		LinQuadDomain.__init__(self, A = A, b = b, lb = lb, ub = ub, A_eq = A_eq, b_eq = b_eq, names = None, **kwargs)

	def _isinside(self, X):
		return self._isinside_bounds(X) & self._isinside_ineq(X) & self._isinside_eq(X)

	def _extent(self, x, p):
		# Check that direction satisfies equality constraints to a tolerance
		if self.A_eq.shape[0] == 0 or np.all(np.abs(self.A_eq.dot(p) ) < self.tol):
			return min(self._extent_bounds(x, p), self._extent_ineq(x, p))
		else:
			return 0. 
	
	def _normalized_domain(self):
		names_norm = [name + ' (normalized)' for name in self.names]
		return LinIneqDomain(lb = self.lb_norm, ub = self.ub_norm, A = self.A_norm, b = self.b_norm, 
			A_eq = self.A_eq_norm, b_eq = self.b_eq_norm, names = names_norm)
 
 
	def chebyshev_center(self):
		r"""Estimates the Chebyshev center using the constrainted least squares approach
	
		Solves the linear program finding the radius :math:`r` and Chebyshev center :math:`\mathbf{x}`. 

		.. math::

			\max_{r\in \mathbb{R}^+, \mathbf{x} \in \mathcal{D}} &\  r \\
			\text{such that} & \ \mathbf{a}_i^\top \mathbf{x} + r \|\mathbf{a}_i\|_2 \le b_i

		where we have expressed the domain in terms of the linear inequality constraints 
		:math:`\mathcal{D}=\lbrace \mathbf{x} : \mathbf{A}\mathbf{x} \le \mathbf{b}\rbrace`
		and :math:`\mathbf{a}_i^\top` are the rows of :math:`\mathbf{A}` as described in [BVNotes]_. 

	
		Returns
		-------
		center: np.ndarray(m,)
			Center of the domain
		radius: float
			radius of largest circle inside the domain

		References
		----------
		.. [BVNotes] https://see.stanford.edu/materials/lsocoee364a/04ConvexOptimizationProblems.pdf, page 4-19.
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
		
		# See p.4-19 https://see.stanford.edu/materials/lsocoee364a/04ConvexOptimizationProblems.pdf
		# 
		normA = np.sqrt( np.sum( np.power(A, 2), axis=1 ) ).reshape((A.shape[0], ))
		
		r = cp.Variable(1)
		x = cp.Variable(len(self))
			
		constraints = [x.__rmatmul__(A) + normA * r <= b]
		if len(self.A_eq) > 0:
			constraints += [x.__rmatmul__(self.A_eq) == self.b_eq]	

		problem = cp.Problem(cp.Maximize(r), constraints)
		problem.solve(**self.kwargs)
		radius = float(r.value)
		center = np.array(x.value).reshape(len(self))
		
		#AA = np.hstack(( A, normA.reshape(-1,1) ))
		#c = np.zeros((A.shape[1]+1,))
		#c[-1] = -1.0
		#A_eq = np.hstack([self.A_eq, np.zeros( (self.A_eq.shape[0],1))])
		#zc = linprog(c, A_ub = AA, b_ub = b, A_eq = A_eq, b_eq = self.b_eq )
		#center = zc[:-1].reshape((n,))
		#radius = zc[-1]
		#print(center)
		#print(self.isinside(center))
		#zc = cp.Variable(len(self)+1)
		#prob = cp.Problem(cp.Maximize(zc[-1]), [zc.__rmatmul__(AA) <= b])
		#prob.solve()
		#print(zc.value[0:-1])

		self._radius = radius
		self._cheb_center = center
		
		return center, radius
		
	@property
	def radius(self):
		try:
			return self._radius
		except:
			self.chebyshev_center()
			return self._radius

	@property
	def center(self):
		try:
			return self._cheb_center
		except:
			self.chebyshev_center()
			return self._cheb_center

	@property
	def Ls(self): return []

	@property
	def ys(self): return []

	@property
	def rhos(self): return []
	


class BoxDomain(LinIneqDomain):
	r""" Implements a domain specified by box constraints

	Given a set of lower and upper bounds, this class defines the domain

	.. math::

		\mathcal{D} := \lbrace \mathbf{x} \in \mathbb{R}^m : \text{lb} \le \mathbf{x} \le \text{ub} \rbrace \subset \mathbb{R}^m.

	Parameters
	----------
	lb: array-like (m,)
		Lower bounds
	ub: array-like (m,)
		Upper bounds
	"""
	def __init__(self, lb, ub, names = None):
		LinQuadDomain.__init__(self, lb = lb, ub = ub, names = names)	
		assert np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)), "Both lb and ub must be finite to construct a box domain"

	# Due to the simplicity of this domain, we can use a more efficient sampling routine
	def _sample(self, draw = 1):
		x_sample = np.random.uniform(self.lb, self.ub, size = (draw, len(self)))
		return x_sample

	def _corner(self, p):
		# Since the domain is a box, we can find the corners simply by looking at the sign of p
		x = np.copy(self.lb)
		I = (p>=0)
		x[I] = self.ub[I]
		return x

	def _extent(self, x, p):
		return self._extent_bounds(x, p)

	def _isinside(self, X):
		return self._isinside_bounds(X)

	def _normalized_domain(self):
		names_norm = [name + ' (normalized)' for name in self.names]
		return BoxDomain(lb = self.lb_norm, ub = self.ub_norm, names = names_norm)

	@property
	def A(self): return np.zeros((0,len(self)))
	
	@property
	def b(self): return np.zeros((0))
	
	@property
	def A_eq(self): return np.zeros((0,len(self)))
	
	@property
	def b_eq(self): return np.zeros((0))
	
	def _isinside(self, X):
		return self._isinside_bounds(X) 

	def _extent(self, x, p):
		return self._extent_bounds(x, p)
	
	def quadrature_rule(self, N, method = 'auto'):
		r""" Constructs quadrature rule for the domain

		Given N, constructs a quadrature rule for $M \le N$ such that 

		.. math::
		
			\int_{\mathbf x\in \mathcal D} f(\mathbb{x}) \mathrm{d}\mathbf{x}
			\approx \sum_{j=1}^M w_j f(\mathbf{x}_j).
		
		If 

		Parameters
		----------
		N: int
			Number of samples to use to construct estimate
		method: ['auto', 'gauss', 'montecarlo']
			Quadrature rule to use


		Returns
		-------
		X: np.ndarray (M, len(self))
			Samples from the domain
		w: np.ndarray (M,)
			Weights for quadrature rule

		"""

		if method == 'auto':
			# The number of samples 
			q = int(np.floor( N**(1./len(self))))
			if q > 1:
				method = 'gauss'
			else:
				method = 'montecarlo'
	
		if method == 'gauss':
			q = int(np.floor( N**(1./len(self))))
			# points in each axis for the quadrature rule
			xs = []
			ws = []
			for i in range(len(self)):
				x, w = gauss(q, self.lb[i], self.ub[i])
				xs.append(x)
				ws.append(w)
		
			# Construct the samples	
			Xs = np.meshgrid(*xs)
			# Flatten into (M, len(self)) shape 
			X = np.hstack([X.reshape(-1,1) for X in Xs])

			# Construct the weights
			Ws = np.meshgrid(*ws)
			W = np.hstack([W.reshape(-1,1) for W in Ws])
			w = np.prod(W, axis = 1)
			return X, w
			

		elif method == 'montecarlo':
			# Call the default Monte-Carlo quadrature rule
			return Domain.quadrature_rule(self, N)	



class PointDomain(BoxDomain):
	r""" A domain consisting of a single point

	Given a point :math:`\mathbf{x} \in \mathbb{R}^m`, construct the domain consisting of that point

	.. math::
	
		\mathcal{D} = \lbrace \mathbf x \rbrace \subset \mathbb{R}^m.

	Parameters
	----------
	x: array-like (m,)
		Single point to be contained in the domain
	"""
	def __init__(self, x, names = None):
		self._point = np.array(x).flatten()
		self._init_names(names)	
		assert len(self._point.shape) == 1, "Must provide a one-dimensional point"

	def __len__(self):
		return self._point.shape[0]
		
	def closest_point(self, x0):
		return np.copy(self._point)

	def _corner(self, p):
		return np.copy(self._point)

	def _extent(self, x, p):
		return 0

	def _isinside(self, X):
		Pcopy = np.tile(self._point.reshape(1,-1), (X.shape[0],1))
		return np.all(X == Pcopy, axis = 1)	

	def _sample(self, draw = 1):
		return np.tile(self._point.reshape(1,-1), (draw, 1))

	@property
	def lb(self):
		return np.copy(self._point)

	@property
	def ub(self):
		return np.copy(self._point)


class ConvexHullDomain(LinIneqDomain):
	r"""Define a domain that is the interior of a convex hull of points.

	Given a set of points :math:`\lbrace x_i \rbrace_{i=1}^M\subset \mathbb{R}^m`,
	construct a domain from their convex hull:

	.. math::
	
		\mathcal{D} := \left\lbrace \sum_{i=1}^M \alpha_i x_i : \sum_{i=1}^M \alpha_i = 1, \ \alpha_i \ge 0 \right\rbrace \subset \mathbb{R}^m.

	In the current implementation, this domain is built by constructing the convex hull of these points
	and then converting it into a linear inequality constrained domain.
	This is expensive for moderate dimensions (>5) and so care should be applied when using this function.

	Parameters
	----------
	X: array-like (M, m)
		Points from which to build the convex hull of points.
	"""
	def __init__(self, X, names = None):
		self.X = np.copy(X)
		if self.X.shape[1] > 1:
			self.hull = ConvexHull(self.X) 
			A = self.hull.equations[:,:-1]
			b = -self.hull.equations[:,-1]
			LinIneqDomain.__init__(self, A, b, names = names)
			self.vertices = np.copy(self.X[self.hull.vertices])
		else:
			lb = np.atleast_1d(np.min(X))
			ub = np.atleast_1d(np.max(X))
			A = np.zeros((0, 1))
			b = np.zeros((0,))
			LinIneqDomain.__init__(self, A, b, lb = lb, ub = ub, names = names)
			self.vertices = np.array([lb, ub]).reshape(-1,1)

class TensorProductDomain(Domain):
	r""" A class describing a tensor product of a multiple domains


	Parameters
	----------
	domains: list of domains
		Domains to combine into a single domain
	"""
	def __init__(self, domains = None):
		self._domains = []
		if domains == None:
			domains = []
	
		for domain in domains:
			assert isinstance(domain, Domain), "Input must be list of domains"
			if isinstance(domain, TensorProductDomain):
				# If one of the sub-domains is a tensor product domain,
				# flatten it to limit recursion
				self._domains.extend(domain.domains)
			else:
				self._domains.append(domain)

	@property
	def names(self):
		return list(itertools.chain(*[dom.names for dom in self.domains]))

	def _is_linquad(self):
		return all([isinstance(dom, LinQuadDomain) for dom in self.domains])
		

	@property
	def domains(self):
		return self._domains

	@property
	def _slices(self):	
		start, stop = 0,0
		for dom in self.domains:
			stop += len(dom)
			yield(slice(start, stop))
			start += len(dom) 	

	def _sample(self, draw = 1):
		X = []
		for dom in self.domains:
			X.append(dom.sample(draw = draw))
		return np.hstack(X)

	def _isinside(self, X):
		inside = np.ones(X.shape[0], dtype = np.bool)
		for dom, I in zip(self.domains, self._slices):
			#print(dom, I, dom.isinside(X[:,I]))
			inside = inside & dom.isinside(X[:,I])
		return inside

	def _extent(self, x, p):
		alpha = [dom._extent(x[I], p[I]) for dom, I in zip(self.domains, self._slices)]
		return min(alpha)


	def __len__(self):
		return sum([len(dom) for dom in self.domains])


	################################################################################		
	# Normalization related functions
	################################################################################		
	
	def _normalize(self, X):
		return np.hstack([dom.normalize(X[:,I]) for dom, I in zip(self.domains, self._slices)])

	def _unnormalize(self, X_norm):
		return np.hstack([dom.unnormalize(X_norm[:,I]) for dom, I in zip(self.domains, self._slices)])

	def _unnormalize_der(self):
		return np.diag(np.hstack([np.diag(dom._unnormalize_der()) for dom in self.domains]))

	def _normalize_der(self):
		return np.diag(np.hstack([np.diag(dom._normalize_der()) for dom in self.domains]))

	def _normalized_domain(self):
		domains_norm = [dom.normalized_domain() for dom in self.domains]
		return TensorProductDomain(domains = domains_norm)

	
	################################################################################		
	# Convex solver problems
	################################################################################		
	
	def _build_constraints_norm(self, x_norm):
		assert self._is_linquad(), "Cannot solve for constraints on the domain"
		constraints = []
		for dom, I in zip(self.domains, self._slices):
			constraints.extend(dom._build_constraints_norm(x_norm[I]))
		return constraints 

	# TODO: There needs to be some way to inherit solver arguments from sub-domains
	def _closest_point(self, x0, L = None, **kwargs):
		return closest_point(self, x0, L = L, **kwargs)

	def _corner(self, p, **kwargs):
		return corner(self, p,  **kwargs)
	
	def _constrained_least_squares(self, A, b, **kwargs):
		return constrained_least_squares(self, A, b,  **kwargs)


	################################################################################		
	# Properties resembling LinQuad Domains 
	################################################################################		

	@property
	def lb(self):
		return np.concatenate([dom.lb for dom in self.domains])

	@property
	def ub(self):
		return np.concatenate([dom.ub for dom in self.domains])

	@property
	def A(self):
		A = []
		for dom, I in zip(self.domain, self._slices):
			A_tmp = np.zeros((dom.A.shape[0] ,len(self)))
			A_tmp[:,I] = dom.A
			A.append(A_tmp)
		return np.vstack(A)

	@property
	def b(self):
		return np.concatenate([dom.b for dom in self.domains])

	@property
	def A_eq(self):
		A_eq = []
		for dom, I in zip(self.domain, self._slices):
			A_tmp = np.zeros((dom.A_eq.shape[0] ,len(self)))
			A_tmp[:,I] = dom.A_eq
			A_eq.append(A_tmp)
		return np.vstack(A_eq)
	
	@property
	def b_eq(self):
		return np.concatenate([dom.b_eq for dom in self.domains])

	

class RandomDomain(Domain):
	r"""Abstract base class for domains with an associated sampling measure
	"""

	def pdf(self, x):
		r""" Probability density function associated with the domain

		This evaluates a probability density function :math:`p:\mathcal{D}\to \mathbb{R}_*`
		at the requested points. By definition, this density function is normalized
		to have measure over the domain to be one:

		.. math::

			\int_{\mathbf{x} \in \mathcal{D}} p(\mathbf{x}) \mathrm{d} \mathbf{x}.

		Parameters
		----------
		x: array-like, either (m,) or (N,m)
			points to evaluate the density function at
	
		Returns
		-------
		array-like (N,)
			evaluation of the density function

		"""
		x = np.array(x)
		if len(x.shape) == 1:
			x = x.reshape(-1,len(self))
			return self._pdf(x).flatten()
		else:
			x = np.array(x).reshape(-1,len(self))
			return self._pdf(x)

	def _pdf(self, x):
		raise NotImplementedError

class UniformDomain(BoxDomain, RandomDomain):
	r""" A randomized version of a BoxDomain with a uniform measure on the space.
	"""
	
	def _pdf(self, x):
		return np.one(x.shape[0])/np.prod([(ub_ - lb_) for lb_, ub_ in zip(self.lb, self.ub)])

class NormalDomain(LinQuadDomain, RandomDomain):
	r""" Domain described by a normal distribution

	This class describes a normal distribution with 
	mean :math:`\boldsymbol{\mu}\in \mathbb{R}^m` and 
	a symmetric positive definite covariance matrix :math:`\boldsymbol{\Gamma}\in \mathbb{R}^{m\times m}`
	that has the probability density function:

	.. math:: 

		p(\mathbf{x}) = \frac{
			e^{-\frac12 (\mathbf{x} - \boldsymbol{\mu}) \boldsymbol{\Gamma}^{-1}(\mathbf{x} - \boldsymbol{\mu})}
			}{\sqrt{(2\pi)^m |\boldsymbol{\Gamma}|}} 

	If the parameter :code:`truncate` is specified, this distribution is truncated uniformly; i.e.,
	calling this parameter :math:`\tau`, the resulting domain has measure :math:`1-\tau`.
	Specifically, if we have a Cholesky decomposition of :math:`\boldsymbol{\Gamma} = \mathbf{L} \mathbf{L}^\top`
	we find a :math:`\rho` such that

	.. math::
		
		\mathcal{D} &= \lbrace \mathbf{x}: \|\mathbf{L}^{-1}(\mathbf{x} - \boldsymbol{\mu})\|_2^2 \le \rho\rbrace ; \\
		p(\mathcal{D}) &= 1-\tau.
		 
	This is done so that the domain has compact support which is necessary for several metric-based sampling strategies.


	Parameters
	----------
	mean : array-like (m,)
		Mean 
	cov : array-like (m,m), optional
		Positive definite Covariance matrix; defaults to the identity matrix
	truncate: float in [0,1), optional
		Amount to truncate the domain to ensure compact support
	"""
	def __init__(self, mean, cov = None, truncate = None, names = None, **kwargs):
		self.tol = 1e-6	
		self.kwargs = kwargs
		######################################################################################	
		# Process the mean
		######################################################################################	
		if isinstance(mean, (int, float)):
			mean = [mean]
		self.mean = np.array(mean)
		self._dimension = m = self.mean.shape[0]
	
		######################################################################################	
		# Process the covariance
		######################################################################################	
		if isinstance(cov, (int, float)):
			cov = np.array([[cov]])
		elif cov is None:
			cov = np.eye(m)
		else:
			cov = np.array(cov)
			assert cov.shape[0] == cov.shape[1], "Covariance must be square"
			assert cov.shape[0] == len(self),  "Covariance must be the same shape as mean"
			assert np.all(np.isclose(cov,cov.T)), "Covariance matrix must be symmetric"
		
		self.cov = cov
		self.L = scipy.linalg.cholesky(self.cov, lower = True)		
		self.Linv = scipy.linalg.solve_triangular(self.L, np.eye(len(self)), lower = True, trans = 'N')
		self.truncate = truncate

		if truncate is not None:
			# Clip corresponds to the 2-norm squared where we should trim based on the truncate
			# parameter.  1 - cdf is the survival function, so we call the inverse survival function to locate
			# this parameter.
			self.clip = scipy.stats.chi2.isf(truncate, len(self)) 
			
			self._Ls = [np.copy(self.Linv) ]
			self._ys = [np.copy(self.mean)]
			# As the transform by Linv places this as a standard-normal,
			# we truncate at the clip parameter.
			self._rhos = [np.sqrt(self.clip)]

		else:
			self.clip = None
			self._Ls = []
			self._ys = []
			self._rhos = []

		self._init_names(names)

	def _sample(self, draw = 1):
		X = np.random.randn(draw, self.mean.shape[0])
		if self.clip is not None:
			# Under the assumption that the truncation parameter is small,
			# we use replacement sampling.
			while True:
				# Find points that violate the clipping
				I = np.sum(X**2, axis = 1) > self.clip
				if np.sum(I) == 0:
					break
				X[I,:] = np.random.randn(np.sum(I), self.mean.shape[0])
		
		# Convert from standard normal into this domain
		X = (self.mean.reshape(-1,1) + self.L.dot(X.T) ).T
		return X


	def _center(self):
		# We redefine the center because due to anisotropy in the covariance matrix,
		# the center is *not* the mean of the coordinate-wise bounds
		return np.copy(self.mean)

	def _normalized_domain(self):
		# We need to do this to keep the sampling measure correct
		names_norm = [name + ' (normalized)' for name in self.names]
		D = self._normalize_der()
		return NormalDomain(self.normalize(self.mean), D.dot(self.cov).dot(D.T), truncate = self.truncate, names = names_norm)

	

	################################################################################		
	# Simple properties
	################################################################################		
	@property
	def lb(self): return -np.inf*np.ones(len(self))
	
	@property
	def ub(self): return np.inf*np.ones(len(self))

	@property
	def A(self): return np.zeros((0,len(self)))

	@property
	def b(self): return np.zeros(0)

	@property
	def A_eq(self): return np.zeros((0,len(self)))

	@property
	def b_eq(self): return np.zeros(0)


	def _isinside(self, X):
		return self._isinside_quad(X) 

	def _pdf(self, X):
		# Mahalanobis distance
		d2 = np.sum(self.Linv.dot(X.T - self.mean.reshape(-1,1))**2, axis = 0)
		# Normalization term
		p = np.exp(-0.5*d2) / np.sqrt((2*np.pi)**len(self) * np.abs(scipy.linalg.det(self.cov)))
		if self.truncate is not None:
			p /= (1-self.truncate)
		return p

# TODO: Ensure sampling is still correct (IMPORTANT FOR DUU Solution)
class LogNormalDomain(BoxDomain, RandomDomain):
	r"""A one-dimensional domain described by a log-normal distribution.

	Given a normal distribution :math:`\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Gamma})`,
	the log normal is described by

	.. math::

		x = \alpha + \beta e^y, \quad y \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Gamma})

	where :math:`\alpha` is an offset and :math:`\beta` is a scaling coefficient. 
	

	Parameters
	----------
	mean: float
		mean of normal distribution feeding the log-normal distribution
	cov: float, optional  
		covariance of normal distribution feeding the log-normal distribution
	offset: float, optional
		Shift the distribution
	scaling: float or np.ndarray
		Scale the output of the log-normal distribution
	truncate: float [0,1)
		Truncate the tails of the distribution
	"""	
	def __init__(self, mean, cov = 1., offset = 0., scaling = 1., truncate = None, names = None):
		self.tol = 1e-6
		self.normal_domain = NormalDomain(mean, cov, truncate = truncate)
		assert len(self.normal_domain) == 1, "Only defined for one-dimensional distributions"

		self.mean = float(self.normal_domain.mean)
		self.cov = float(self.normal_domain.cov)
		self.scaling = float(scaling)
		self.offset = float(offset)
		self.truncate = truncate

		# Determine bounds
		# Because this doesn't have a convex relationship to the multivariate normal
		# truncated domains, we manually specify these here as they cannot be inferred
		# from the (non-existant) quadratic constraints as in the NormalDomain case.
		if self.truncate is not None:
			self._lb = self.offset + self.scaling*np.exp(self.normal_domain.norm_lb)
			self._ub = self.offset + self.scaling*np.exp(self.normal_domain.norm_ub)
		else:
			self._lb = 0.*np.ones(1)
			self._ub = np.inf*np.ones(1)

		self._init_names(names)

	def __len__(self):
		return len(self.normal_domain)

	def _sample(self, draw = 1):
		X = self.normal_domain.sample(draw)
		return np.array(self.offset).reshape(-1,1) + self.scaling*np.exp(X)

	def _normalized_domain(self):
		names_norm = [name + ' (normalized)' for name in self.names]
		if self.truncate is not None:
			c = self._center()
			D = float(self._normalize_der()) 
			return LogNormalDomain(self.normal_domain.mean, self.normal_domain.cov, 
				offset = D*(self.offset - c) , scaling = D*self.scaling, truncate = self.truncate, names = names_norm)
		else:
			return self

	def _pdf(self, X):
		X_norm = (X - self.offset)/self.scaling
		p = np.exp(-(np.log(X_norm) - self.mean)**2/(2*self.cov))/(X_norm*self.cov*np.sqrt(2*np.pi))
		return p


if __name__ == '__main__':
	dom = BoxDomain([-1,-1], [1,1])
	dom.quadrature_rule(100)
