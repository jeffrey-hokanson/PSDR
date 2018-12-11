"""Base domain types"""
from __future__ import print_function, division

import numpy as np
from scipy.optimize import newton, brentq
from copy import deepcopy

from scipy.optimize import nnls, minimize
from scipy.linalg import orth
from scipy.spatial import ConvexHull

from opt import *
#from opt_gurobi import linobj_gurobi
import cvxpy as cp
import warnings

__all__ = ['Domain',
		'LinQuadDomain',
		'LinIneqDomain',
		'ConvexHullDomain',
		'BoxDomain', 
		'PointDomain',
		'UniformDomain',
		'NormalDomain',
		'LogNormalDomain',
		'TensorProductDomain',
	] 


class EmptyDomain(Exception):
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
		obj = x_norm.__rmatmul__(D.dot(p).reshape(1,-1))
		constraints = dom._build_constraints_norm(x_norm)
		problem = cp.Problem(cp.Maximize(obj), constraints)
		problem.solve(**kwargs)
	return dom.unnormalize(np.array(x_norm.value).reshape(len(dom)))




class Domain(object):
	r""" Abstract base class for an input domain

	This specifies a domain :math:`\mathcal{D}\subset \mathbb{R}^m`.

	"""
	
	def __len__(self):
		raise NotImplementedError

	# To define the documentation once for all domains, these functions call internal functions
	# to each subclass
	
	def closest_point(self, x0, L = None, **kwargs):
		"""Given a point, find the closest point in the domain to it.

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
		"""Compute the distance alpha such that x + alpha * p is on the boundary of the domain

		Given a point :math:`\mathbf{x}\in\mathcal D` and a direction :math:`\mathbf p`,
		find the furthest we can go in direction :math:`\mathbf p` and stay inside the domain:

		.. math::
	
			\max_{\\alpha > 0}   \\alpha \quad \\text{such that} \quad \mathbf x + \\alpha \mathbf p \in \mathcal D

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
		if len(X.shape) == 1:
			X = X.reshape(-1, len(self)) 	
			return self._isinside(X).flatten()
		else:
			return self._isinside(X)



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
			self._hit_and_run_state = x0	

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


	def __mul__(self, other):
		r""" Create a tensor product domain
		"""
		return TensorProductDomain([self, other])

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
		return [y - L.dot(c) for L, y in zip(self.Ls, self.ys)]	

	@property
	def rhos_norm(self):
		return self.rhos
	
	################################################################################		
	# Normalization functions 
	################################################################################		

	def isnormalized(self):
		return np.all( (~np.isfinite(self.lb)) | (self.lb == -1.) ) and np.all( (~np.isfinite(self.ub)) | (self.ub == 1.) ) 

	def _normalize_der(self):
		"""Derivative of normalization function"""
		slope = np.ones(len(self))
		I = (self.ub != self.lb) & np.isfinite(self.lb) & np.isfinite(self.ub)
		slope[I] = 2.0/(self.ub[I] - self.lb[I])
		return np.diag(slope)

	def _unnormalize_der(self):
		slope = np.ones(len(self))
		I = (self.ub != self.lb) & np.isfinite(self.lb) & np.isfinite(self.ub)
		slope[I] = (self.ub[I] - self.lb[I])/2.0
		return np.diag(slope)
	
	def _center(self):
		c = np.zeros(len(self))
		I = np.isfinite(self.lb) & np.isfinite(self.ub)
		c[I] = (self.lb[I] + self.ub[I])/2.0
		return c	

	def _normalize(self, X):
		# reshape so numpy's broadcasting works correctly
		lb = self.lb.reshape(1, -1)
		ub = self.ub.reshape(1, -1)
		
		# Those points with zero range get mapped to zero, so we only work on those
		# with a non-zero range
		X_norm = np.zeros(X.shape)
		I = (self.ub != self.lb) & np.isfinite(self.lb) & np.isfinite(self.ub)
		X_norm[:,I] = 2.0 * (X[:,I] - lb[:,I]) / (ub[:,I] - lb[:,I]) - 1.0
		#I = (self.ub != self.lb)
		#X_norm[:,I] = 0
		I = ~np.isfinite(self.lb) | ~np.isfinite(self.ub)
		X_norm[:,I] = X[:,I]
		return X_norm
	
	def _unnormalize(self, X_norm, **kwargs):
		# reshape so numpy's broadcasting works correctly
		lb = self.lb.reshape(1, -1)
		ub = self.ub.reshape(1, -1)
		
		# Idenify parameters with nonzero range
		X = np.zeros(X_norm.shape)

		# unnormalize parameters with non-zero range and bounded
		I = (self.ub != self.lb) & np.isfinite(self.lb) & np.isfinite(self.ub)
		X[:,I] = (ub[:,I] - lb[:,I]) * (X_norm[:,I] + 1.0)/2.0 + lb[:,I]
	
		# for those with infinite bounds, apply no transformation
		I = ~np.isfinite(self.lb) | ~np.isfinite(self.ub) 
		X[:,I] = X_norm[:,I]	
		# for those dimensions with zero dimension, set to the center point lb[:,~I] = ub[:,~I]
		I = (self.ub == self.lb)
		X[:,I] = lb[:,I]
		return X 

	################################################################################		
	# Bound checking
	################################################################################		
	def _isinside_bounds(self, X, tol = 1e-10):
		lb_check = np.array([np.all(x >= self.lb-tol) for x in X], dtype = np.bool)
		ub_check = np.array([np.all(x <= self.ub+tol) for x in X], dtype = np.bool)
		return lb_check & ub_check

	def _isinside_ineq(self, X, tol = 1e-10):
		return np.array([np.all(np.dot(self.A, x) <= self.b + tol) for x in X], dtype = np.bool)

	def _isinside_eq(self, X, tol = 1e-10):
		return np.array([np.all( np.abs(np.dot(self.A_eq, x) - self.b_eq) < tol) for x in X], dtype = np.bool)

	def _isinside_quad(self, X, tol = 1e-10):
		"""check that points are inside quadratic constraints"""
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
		**kwargs):

		self.tol = 1e-10
		# Determine dimension of space
		self._init_dim(lb = lb, ub = ub, A = A, A_eq = A_eq, Ls = Ls)

		# Start setting default values
		self._lb = self._init_lb(lb)
		self._ub = self._init_ub(ub)
		self._A, self._b = self._init_ineq(A, b)
		self._A_eq, self._b_eq = self._init_eq(A_eq, b_eq)	
		self._Ls, self._ys, self._rhos = self._init_quad(Ls, ys, rhos)

		if len(kwargs) == 0:
			kwargs= {'solver': cp.CVXOPT, 'reltol': 1e-10, 'abstol' : 1e-10, 'verbose': True}
		self.kwargs = kwargs
	
	################################################################################		
	# Initialization helpers 
	################################################################################		
	def _init_dim(self, lb = None, ub = None, A = None, A_eq = None, Ls = None):
		"""determine the dimension of the space we are working on"""
		if lb is not None:
			m = len(lb)
		elif ub is not None:
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
			assert len(lb) == len(self), "Lower bound has wrong dimensions"
			return np.array(lb)
		
	def _init_ub(self, ub):
		if ub is None:
			return np.inf*np.ones(len(self))
		else:
			assert len(ub) == len(self), "Upper bound has wrong dimensions"
			return np.array(ub)
		
	def _init_ineq(self, A, b):
		if A is None and b is None:
			A = np.zeros((0,len(self)))
			b = np.zeros((0,))
		elif A is not None and b is not None:
			A = np.array(A)
			b = np.array(b)
			assert A.shape[1] == len(self), "A has wrong number of columns"
			assert A.shape[0] == b.shape[0], "The number of rows of A and b do not match"
			assert len(b.shape) == 1, "b must have only one dimension"
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
			assert A_eq.shape[1] == len(self), "A_eq has wrong number of columns"
			assert A_eq.shape[0] == b_eq.shape[0], "The number of rows of A_eq and b_eq do not match"
			assert len(b_eq.shape) == 1, "b_eq must have only one dimension"
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
		return LinQuadDomain(lb = self.lb_norm, ub = self.ub_norm, A = self.A_norm, b = self.b_norm, 
			A_eq = self.A_eq_norm, b_eq = self.b_eq_norm, Ls = self.Ls_norm, ys = self.ys_norm, rhos = self.rhos_norm)
	

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
			constraints.append( cp.norm(x_norm.__rmatmul__(L) - L.dot(y)) <= rho )

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

	Raises
	------
	EmptyDomain:
		raised if cannot find a point inside the domain; can be caused by scaling of constraints.

	"""
	def __init__(self, A = None, b = None, lb = None, ub = None, A_eq = None, b_eq = None, **kwargs):
		LinQuadDomain.__init__(self, A = A, b = b, lb = lb, ub = ub, A_eq = A_eq, b_eq = b_eq, **kwargs)

	def _isinside(self, X):
		return self._isinside_bounds(X) & self._isinside_ineq(X) & self._isinside_eq(X)

	def _extent(self, x, p):
		# Check that direction satisfies equality constraints to a tolerance
		if self.A_eq.shape[0] == 0 or np.all(np.abs(self.A_eq.dot(p) ) < self.tol):
			return min(self._extent_bounds(x, p), self._extent_ineq(x, p))
		else:
			return 0. 
	
	def _normalized_domain(self):
		return LinIneqDomain(lb = self.lb_norm, ub = self.ub_norm, A = self.A_norm, b = self.b_norm, 
			A_eq = self.A_eq_norm, b_eq = self.b_eq_norm)
 
 
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
		
		# See p.4-19 https://see.stanford.edu/materials/lsocoee364a/04ConvexOptimizationProblems.pdf
		# 
		normA = np.sqrt( np.sum( np.power(A, 2), axis=1 ) ).reshape((A.shape[0], 1))
		AA = np.hstack(( A, normA ))
		c = np.zeros((A.shape[1]+1,))
		c[-1] = -1.0
		A_eq = np.hstack([self.A_eq, np.zeros( (self.A_eq.shape[0],1))])
		zc = linprog(c, A_ub = AA, b_ub = b, A_eq = A_eq, b_eq = self.b_eq )

		center = zc[:-1].reshape((n,))
		radius = zc[-1]
		
		self._radius = radius
		self._center = center
		
	@property
	def radius(self):
		try:
			return self._radius
		except:
			self._chebyshev_center()
			return self._radius

	@property
	def center(self):
		try:
			return self._center
		except:
			self._chebyshev_center()
			return self._center


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
	def __init__(self, lb, ub):
		LinQuadDomain.__init__(self, lb = lb, ub = ub)	
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
		return BoxDomain(lb = self.lb_norm, ub = self.ub_norm)


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
	def __init__(self, x):
		self._point = np.array(x).flatten()
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

class TensorProductDomain(Domain):
	r""" A class describing a tensor product of a multiple domains

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
			inside = inside & dom.isinside(X[:,I])
		return inside

	def _extent(self, x, p):
		alpha = [dom.extent(x[I], p[I]) for dom, I in zip(self.domains, self._slices)]
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


	

#	def add_constraint(self, **kwargs):
#		if all([isinstance(dom, LinIneqDomain) for dom in self.domains]):
#			dom = LinIneqDomain(A = self.A, b = self.b, lb = self.lb, ub = self.ub, A_eq = self.A_eq, b_eq = self.b_eq, center = self.center)
#			return dom.add_constraint(**kwargs)
#		else:
#			raise NotImplementedError("We currently do not support adding constraints to TensorProductDomains that are not linear inequality domains")






class RandomDomain(Domain):
	r"""Abstract base class for domains with an associated sampling measure
	"""

	def pdf(self, x):
		r""" Probability density function associated with the domain
		"""
		x = np.array(x).reshape(-1,len(self))
		return self._pdf(self, x)

	def _pdf(self, x):
		raise NotImplementedError

class UniformDomain(BoxDomain, RandomDomain):
	r""" A randomized version of a BoxDomain
	"""
	
	def _pdf(self, x):
		return np.one(x.shape[0])/np.prod([(ub_ - lb_) for lb_, ub_ in zip(self.lb, self.ub)])

class NormalDomain(LinQuadDomain, RandomDomain):
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
		print("called isinside")
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
			print("called extent")
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
			print("invalid value", X/self.scaling)
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
		print("called LogNormalDomain isinside")
		if self.clip is None:
			return np.min(X>0, axis = 1)
		else:
			X_norm = self._normalize_nonlinear(X)
			X2 = np.sqrt(np.sum(X_norm**2, axis = 1))
			return X2 < self.clip	

	@property
	def center(self):
		return self.scaling * np.exp(self.mean.reshape(-1))


