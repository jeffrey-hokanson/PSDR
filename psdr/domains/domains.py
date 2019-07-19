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
from scipy.stats import ortho_group
from scipy.spatial.distance import pdist
import cvxpy as cp

import sobol_seq

TOL = 1e-5

from ..quadrature import *
from ..misc import *
from ..exceptions import *

from .domain import *
from .euclidean import EuclideanDomain

__all__ = [
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

# NOTE: These three functions need to be defined outside of the classes
# so we can call them on both LinQuadDomains and TensorProductDomains,
# the latter of which may not necessarily be a LinQuadDomain.

def closest_point(dom, x0, L, **kwargs):
	r""" Solve the closest point problem given a domain
	"""

	if dom.isinside(x0):
		return np.copy(x0)

	x_norm = cp.Variable(len(dom))
	constraints = dom._build_constraints_norm(x_norm)
	x0_norm =  dom.normalize(x0)
	
	if L is None:
		L = np.eye(len(dom))
		
	D = dom._unnormalize_der() 	
	LD = L.dot(D)
	obj = cp.norm(LD*x_norm - LD.dot(x0_norm))

	problem = cp.Problem(cp.Minimize(obj), constraints)
	problem.solve(**kwargs)

	# TODO: Check solution state 			
	return dom.unnormalize(np.array(x_norm.value).reshape(len(dom)))	

def constrained_least_squares(dom, A, b, **kwargs):
	x_norm = cp.Variable(len(dom))
	D = dom._unnormalize_der() 
	c = dom._center()	
		
	# \| A x - b\|_2 
	obj = cp.norm(x_norm.__rmatmul__(A.dot(D)) - b - A.dot(c) )
	constraints = dom._build_constraints_norm(x_norm)
	problem = cp.Problem(cp.Minimize(obj), constraints)
	problem.solve(**kwargs)
	return dom.unnormalize(np.array(x_norm.value).reshape(len(dom)))

def corner(dom, p, **kwargs):
	# If we already know the domain is empty, error early
	try:
		if dom._empty:
			raise EmptyDomainException
	except AttributeError:
		pass

	# Find the corner using CVXPY

	local_kwargs = merge(dom.kwargs, kwargs)		
	x_norm = cp.Variable(len(dom))
	D = dom._unnormalize_der() 	
		
	# p.T @ x
	if len(dom) > 1:
		obj = x_norm.__rmatmul__(D.dot(p).reshape(1,-1))
	else:
		obj = x_norm*float(D.dot(p))

	constraints = dom._build_constraints_norm(x_norm)
	problem = cp.Problem(cp.Maximize(obj), constraints)
	
	problem.solve(**local_kwargs)

	if problem.status in ['infeasible']:
		dom._empty = True
		raise EmptyDomainException	
	elif problem.status in ['unbounded']:
		dom._unbounded = True
		raise UnboundedDomainException
	elif problem.status not in ['optimal', 'optimal_inaccurate']:
		print(problem.status)
		raise SolverError

	# If we have found a solution, then the domain is not empty
	dom._empty = False
	return dom.unnormalize(np.array(x_norm.value).reshape(len(dom)))




		

	


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

	def _isinside(self, X, tol = TOL):
		Pcopy = np.tile(self._point.reshape(1,-1), (X.shape[0],1))
		return np.all(X == Pcopy, axis = 1)	

	def _sample(self, draw = 1):
		return np.tile(self._point.reshape(1,-1), (draw, 1))

	def is_point(self):
		return True

	@property
	def lb(self):
		return np.copy(self._point)

	@property
	def ub(self):
		return np.copy(self._point)


class ConvexHullDomain(LinQuadDomain):
	r"""Define a domain that is the interior of a convex hull of points.

	Given a set of points :math:`\lbrace x_i \rbrace_{i=1}^M\subset \mathbb{R}^m`,
	construct a domain from their convex hull:

	.. math::
	
		\mathcal{D} := \left\lbrace \sum_{i=1}^M \alpha_i x_i : \sum_{i=1}^M \alpha_i = 1, \ \alpha_i \ge 0 \right\rbrace \subset \mathbb{R}^m.


	Parameters
	----------
	X: array-like (M, m)
		Points from which to build the convex hull of points.
	"""

	def __init__(self, X, A = None, b = None, lb = None, ub = None, 
		A_eq = None, b_eq = None, Ls = None, ys = None, rhos = None,
		names = None, **kwargs):

		self._X = np.copy(X)
		if len(self._X.shape) == 1:
			self._X = self._X.reshape(-1,1)
	
		self._init_names(names)
		

		# Start setting default values
		self._lb = self._init_lb(lb)
		self._ub = self._init_ub(ub)
		self._A, self._b = self._init_ineq(A, b)
		self._A_eq, self._b_eq = self._init_eq(A_eq, b_eq)	
		self._Ls, self._ys, self._rhos = self._init_quad(Ls, ys, rhos)

		# Setup the lower and upper bounds to improve conditioning
		# when solving LPs associated with domain features
		# TODO: should we consider reducing dimension via rotation
		# if the points are 
		self._norm_lb = np.min(self._X, axis = 0)
		self._norm_ub = np.max(self._X, axis = 0)
		self._X_norm = self.normalize(X)
		
		self.kwargs = merge(DEFAULT_CVXPY_KWARGS, kwargs)

	def __str__(self):
		ret = "<ConvexHullDomain on R^%d based on %d points" % (len(self), len(self._X_norm))
		if len(self._Ls) > 0:
			ret += "; %d quadratic constraints" % (len(self._Ls),)
		if self._A.shape[0] > 0:
			ret += "; %d linear inequality constraints" % (self._A.shape[0], )
		if self._A_eq.shape[0] > 0:
			ret += "; %d linear equality constraints" % (self._A_eq.shape[0], )
		
		ret += ">"
		return ret

	def to_linineq(self, **kwargs):
		r""" Convert the domain into a LinIneqDomain

		"""
		if len(self) > 1:
			hull = ConvexHull(self._X) 
			A = hull.equations[:,:-1]
			b = -hull.equations[:,-1]
			dom = LinIneqDomain(A = A, b = b, names = self.names, **kwargs)
			dom.vertices = np.copy(self._X[hull.vertices])
		else:
			dom = BoxDomain(self._lb, self._ub, names = names, **kwargs)

		return dom

	def coefficients(self, x, **kwargs):
		r""" Find the coefficients of the convex combination of elements in the space yielding x

		"""
		x_norm = self.normalize(x)		
		A = np.vstack([self._X_norm.T, np.ones( (1,len(self._X_norm)) )])
		b = np.hstack([x_norm, 1])
		alpha, rnorm = nnls(A, b)
		#print('rnorm', rnorm)
		#assert rnorm < 1e-5, "Point x must be inside the domain"
		return alpha

	@property
	def X(self):
		return np.copy(self._X)

	def __len__(self):
		return self._X.shape[1]
	
	def _closest_point(self, x0, L = None, **kwargs):

		if self.isinside(x0):
			return np.copy(x0)

		if L is None:
			L = np.eye(len(self))
			
		D = self._unnormalize_der() 	
		LD = L.dot(D)
		
		m = len(self)
		x0_norm = self.normalize(x0)
		x_norm = cp.Variable(m)					# Point inside the domain
		alpha = cp.Variable(len(self._X))		# convex combination parameters
		
		obj = cp.Minimize(cp.norm(LD*x_norm - LD.dot(x0_norm) ))
		constraints = [x_norm == alpha.__rmatmul__(self._X_norm.T), alpha >=0, cp.sum(alpha) == 1]
		constraints += self._build_constraints_norm(x_norm)
	
		prob = cp.Problem(obj, constraints)
		prob.solve(**merge(self.kwargs, kwargs))

		return self.unnormalize(np.array(x_norm.value).reshape(len(self)))
	
	def _corner(self, p, **kwargs):
		D = self._unnormalize_der()
		x_norm = cp.Variable(len(self))			# Point inside the domain
		alpha = cp.Variable(len(self._X))		# convex combination parameters
		obj = cp.Maximize(x_norm.__rmatmul__( D.dot(p)))
		constraints = [x_norm == alpha.__rmatmul__(self._X_norm.T), alpha >=0, cp.sum(alpha) == 1]
		constraints += self._build_constraints_norm(x_norm)
		prob = cp.Problem(obj, constraints)
		prob.solve(**merge(self.kwargs, kwargs))
		return self.unnormalize(np.array(x_norm.value).reshape(len(self)))
	
	def _extent(self, x, p, **kwargs):
		kwargs['warm_start'] = True
		if not hasattr(self, '_extent_alpha'):
			self._extent_alpha = cp.Variable(len(self._X))	# convex combination parameters
			self._extent_beta = cp.Variable(1)				# Step length
			self._extent_x_norm = cp.Parameter(len(self))	# starting point inside the domain
			self._extent_p_norm = cp.Parameter(len(self))
			self._extent_obj = cp.Maximize(self._extent_beta)
			self._extent_constraints = [
					self._extent_alpha.__rmatmul__(self._X_norm.T) == self._extent_beta * self._extent_p_norm + self._extent_x_norm,
					self._extent_alpha >=0, 
					cp.sum(self._extent_alpha) == 1,
					]
			self._extent_constraints += self._build_constraints_norm(self._extent_x_norm)
			self._extent_prob = cp.Problem(self._extent_obj, self._extent_constraints)
		
		self._extent_x_norm.value = self.normalize(x)
		self._extent_p_norm.value = self.normalize(x+p)-self.normalize(x)
		self._extent_prob.solve(**merge(self.kwargs, kwargs))
		try:
			return float(self._extent_beta.value)
		except:
			# If we can't solve the problem, we are outside the domain or cannot move futher;
			# hence we return 0
			return 0.		

	def _isinside(self, X, tol = TOL):

		# Check that the points are in the convex hull
		inside = np.zeros(X.shape[0], dtype = np.bool)
		for i, xi in enumerate(X):
			alpha = self.coefficients(xi)
			rnorm = np.linalg.norm( xi - self._X.T.dot(alpha))
			inside[i] = (rnorm < tol)

		# Now check linear inequality/equality constraints and quadratic constraints
		inside &= self._isinside_bounds(X, tol = tol)
		inside &= self._isinside_ineq(X, tol = tol)	
		inside &= self._isinside_eq(X, tol = tol)
		inside &= self._isinside_quad(X, tol = tol)

		return inside


class TensorProductDomain(EuclideanDomain):
	r""" A class describing a tensor product of a multiple domains


	Parameters
	----------
	domains: list of domains
		Domains to combine into a single domain
	**kwargs
		Additional keyword arguments to pass to CVXPY
	"""
	def __init__(self, domains = None, **kwargs):
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
		self.kwargs = merge(DEFAULT_CVXPY_KWARGS, kwargs)

	def __str__(self):
		return "<TensorProductDomain on R^%d of %d domains>" % (len(self), len(self.domains))
		
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

	def _isinside(self, X, tol = TOL):
		inside = np.ones(X.shape[0], dtype = np.bool)
		for dom, I in zip(self.domains, self._slices):
			#print(dom, I, dom.isinside(X[:,I]))
			inside = inside & dom.isinside(X[:,I], tol = tol)
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

	def _normalized_domain(self, **kwargs):
		domains_norm = [dom.normalized_domain(**kwargs) for dom in self.domains]
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
		if self._is_linquad():
			return closest_point(self, x0, L = L, **kwargs)
		else:
			raise NotImplementedError("Cannot find closest point on domains that are not LinQuadDomains")

	def _corner(self, p, **kwargs):
		if self._is_linquad():
			return corner(self, p, **kwargs)
		else:
			raise NotImplementedError("Cannot find corners on domains that are not LinQuadDomains")
	
	def _constrained_least_squares(self, A, b, **kwargs):
		if self._is_linquad():
			return constrained_least_squares(self, A, b, **kwargs)
		else:
			raise NotImplementedError("Cannot solve constrained least squares problems on domains that are not LinQuadDomains")


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

	

class RandomDomain(EuclideanDomain):
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
		self.kwargs = merge(DEFAULT_CVXPY_KWARGS, kwargs)
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

	def _normalized_domain(self, **kwargs):
		# We need to do this to keep the sampling measure correct
		names_norm = [name + ' (normalized)' for name in self.names]
		D = self._normalize_der()
		return NormalDomain(self.normalize(self.mean), D.dot(self.cov).dot(D.T), truncate = self.truncate, names = names_norm, 
			**merge(self.kwargs, kwargs))

	

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


	def _isinside(self, X, tol = TOL):
		return self._isinside_quad(X, tol = tol) 

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

	def _normalized_domain(self, **kwargs):
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

