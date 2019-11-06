from __future__ import division

import numpy as np
import cvxpy as cp

from scipy.optimize import nnls
from scipy.spatial import ConvexHull


from .domain import TOL, DEFAULT_CVXPY_KWARGS
from .linquad import LinQuadDomain
from .box import BoxDomain
from ..misc import merge


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
	names: list of strings, optional
		Names for each of the parameters in the space
	kwargs: dict, optional
		Additional parameters to be passed to cvxpy Problem.solve() 
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

	def _is_box_domain(self):
		return False

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
			dom_hull = LinQuadDomain(A = A, b = b, names = self.names, **kwargs)
			dom_hull.vertices = np.copy(self._X[hull.vertices])
			dom = dom_hull.add_constraints(A = self.A, b = self.b, A_eq = self.A_eq, b_eq = self.b_eq,
				Ls = self.Ls, ys = self.ys, rhos = self.rhos)
		else:
			lb = self.corner([-1])
			ub = self.corner([1])
			dom = BoxDomain(lb, ub, names = self.names, **kwargs)

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


	def _build_constraints(self, x):
		
		alpha = cp.Variable(len(self.X), name = 'alpha')
		constraints = [x_norm == alpha.__rmatmul__(self._X.T), alpha >=0, cp.sum(alpha) == 1]
		constraints += LinQuadDomain._build_constraints(self, x_norm)
		return constraints
		
	def _build_constraints_norm(self, x_norm):
		alpha = cp.Variable(len(self.X), name = 'alpha')
		constraints = [x_norm == alpha.__rmatmul__(self._X_norm.T), alpha >=0, cp.sum(alpha) == 1]
		constraints += LinQuadDomain._build_constraints_norm(self, x_norm)
		return constraints
	
#	def _closest_point(self, x0, L = None, **kwargs):
#
#		if self.isinside(x0):
#			return np.copy(x0)
#
#		if L is None:
#			L = np.eye(len(self))
#			
#		D = self._unnormalize_der() 	
#		LD = L.dot(D)
#		
#		m = len(self)
#		x0_norm = self.normalize(x0)
#		x_norm = cp.Variable(m)					# Point inside the domain
#		alpha = cp.Variable(len(self._X))		# convex combination parameters
#		
#		obj = cp.Minimize(cp.norm(LD*x_norm - LD.dot(x0_norm) ))
#		constraints = [x_norm == alpha.__rmatmul__(self._X_norm.T), alpha >=0, cp.sum(alpha) == 1]
#		constraints += LinQuadDomain._build_constraints_norm(self, x_norm)
#		#constraints += self._build_constraints_norm(x_norm)
#	
#		prob = cp.Problem(obj, constraints)
#		prob.solve(**merge(self.kwargs, kwargs))
#
#		return self.unnormalize(np.array(x_norm.value).reshape(len(self)))
#	
#	def _corner(self, p, **kwargs):
#		D = self._unnormalize_der()
#		x_norm = cp.Variable(len(self))			# Point inside the domain
#		alpha = cp.Variable(len(self._X))		# convex combination parameters
#		obj = cp.Maximize(x_norm.__rmatmul__( D.dot(p)))
#		constraints = [x_norm == alpha.__rmatmul__(self._X_norm.T), alpha >=0, cp.sum(alpha) == 1]
#		#constraints += self._build_constraints_norm(x_norm)
#		constraints += LinQuadDomain._build_constraints_norm(self, x_norm)
#		prob = cp.Problem(obj, constraints)
#		prob.solve(**merge(self.kwargs, kwargs))
#		return self.unnormalize(np.array(x_norm.value).reshape(len(self)))
	
	def _extent(self, x, p, **kwargs):
		# NB: We setup cached description of this problem because it is used repeatedly
		# when hit and run sampling this domain
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
			#self._extent_constraints += self._build_constraints_norm(self._extent_x_norm)
			self._extent_constraints += LinQuadDomain._build_constraints_norm(self, self._extent_x_norm)
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
