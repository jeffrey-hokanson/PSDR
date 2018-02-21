"""Base domain types"""

import numpy as np
from scipy.optimize import newton, brentq
from copy import deepcopy

from scipy.optimize import nnls, minimize
from scipy.linalg import orth
from scipy.spatial import ConvexHull

from util import linprog

__all__ = ['Domain', 'ComboDomain', 'BoxDomain', 'UniformDomain', 'NormalDomain', 'LogNormalDomain', 'LinIneqDomain', 'ConvexHullDomain'] 

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
	""" Abstract base class for input domain
	"""	
	def sample(self, draw = 1):
		""" Generate a random sample from the domain
		
		Parameters
		----------
		draw : positive integer
			Number of sample points to draw
		"""
		raise NotImplementedError
		
	def isinside(self, x):
		"""Determines if a point is inside the domain

		"""
		raise NotImplementedError


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
		raise NotImplementedError

	def normalize(self, x):
		""" Given a point in the application space, convert it to normalized units
		"""
		raise NotImplementedError
	
	def unnormalize(self, x):
		raise NotImplementedError

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

	def dist(self, x, y):
		x = self.normalize(x)
		y = self.normalize(y)
		return np.linalg.norm(x - y)**2


	def range_norm(self, U_norm):
		""" Compute range along U_norm in the normalized space 
		"""
		U_norm = U_norm.flatten()
		assert U_norm.shape[0] == self.center.shape[0], "U has wrong dimensions"
		xp = linprog(-U_norm, A_ub = self.A_norm, b_ub = self.b_norm, lb = self.lb_norm, ub = self.ub_norm)
		xn = linprog(U_norm, A_ub = self.A_norm, b_ub = self.b_norm, lb = self.lb_norm, ub = self.ub_norm)
		min_range = np.dot(U_norm.T, xn)
		max_range = np.dot(U_norm.T, xp)
		return np.sort([min_range, max_range])
	
	def range(self, U):
		""" Compute range along U in the application space 
		"""
		U = U.flatten()
		assert U.shape[0] == self.center.shape[0], "U has wrong dimensions"
		xp = linprog(-U, A_ub = self.A, b_ub = self.b, lb = self.lb, ub = self.ub)
		xn = linprog(U, A_ub = self.A, b_ub = self.b, lb = self.lb, ub = self.ub)
		min_range = np.dot(U.T, xn)
		max_range = np.dot(U.T, xp)
		return np.sort([min_range, max_range])

	def corner_norm(self, p_norm):
		""" find the point furthest along the direction p in the domain
		"""
		x_norm = linprog(-p_norm, 
			A_ub = self.A_norm,
			b_ub = self.b_norm,
			lb = self.lb_norm,
			ub = self.ub_norm,
			A_eq = self.A_eq_norm,
			b_eq = self.b_eq_norm,
			)
		return x_norm
	
	def corner(self, p):
		""" find the point furthest along the direction p in the domain
		"""
		x = linprog(-p, 
			A_ub = self.A,
			b_ub = self.b,
			lb = self.lb,
			ub = self.ub,
			A_eq = self.A_eq,
			b_eq = self.b_eq,
			)
		return x

	def build_equality_domain_norm(self, U_norm, y_norm):
		""" Build a domain that specifies a particular active coordinate

		"""
		y_norm = np.atleast_1d(y_norm)

		# To initialize the center, we use the fact that the domain is convex
		xp = linprog(-U_norm.flatten(), A_ub = self.A_norm, b_ub = self.b_norm, lb = self.lb_norm, ub = self.ub_norm)
		xn = linprog(U_norm.flatten(), A_ub = self.A_norm, b_ub = self.b_norm, lb = self.lb_norm, ub = self.ub_norm)
		ap = float(np.dot(U_norm.T, xp))
		an = float(np.dot(U_norm.T, xn))
		assert y_norm >= an	and y_norm <= ap, "y_norm not in feasible domain"
		alpha = (float(y_norm) - an)/(ap - an)
		center = xn*(1-alpha) + xp*alpha
		# Check for 2d U_norm
		return LinIneqDomain(self.A_norm, self.b_norm, lb = self.lb_norm, ub = self.ub_norm, A_eq = U_norm.T, b_eq = y_norm, center = center) 

	def normalized_domain(self):
		raise NotImplementedError


	def ridge_sample_norm(self, U_norm, n_points = 10, samples_per_point = 10):
		"""
		Parameters
		----------
		U_norm : np.array
			Coordinates of ridge subspace in the normalized domain
		n_points : int
			Number of points to sample along the ridge direction, linearly spaced
		samples_per_point : int
			Number of samples at each point along the ridge direction to take
		
		Returns
		-------
		np.array : 
			Points in the original space
		"""  

		# Determine the corners of the box
		xn = linprog(U_norm.flatten(), 
			A_ub = self.A_norm, 
			b_ub = self.b_norm, 
			lb = self.lb_norm, 
			ub = self.ub_norm)
		xp = linprog(-U_norm.flatten(), 
			A_ub = self.A_norm, 
			b_ub = self.b_norm, 
			lb = self.lb_norm, 
			ub = self.ub_norm)

		# Determine how far along the ridge direction these points are
		ap = float(np.dot(U_norm.T, xp))
		an = float(np.dot(U_norm.T, xn))

		# Points along the ridge to sample
		ys = np.linspace(an, ap, n_points + 2)[1:-1]
		X = []
		for y in ys:
			alpha = (y - an)/(ap - an)
			it = 0
			while it < samples_per_point:
				# Pick a random direction
				p = np.random.randn(len(self))

				# Run to the boundary in this direction
				x_norm = linprog(p, 
					A_ub = self.A_norm,
					b_ub = self.b_norm,
					lb = self.lb_norm,
					ub = self.ub_norm,
					A_eq = U_norm.T,
					b_eq = y)
			
				x = self.unnormalize(x_norm)
				
				# Ensure that the point is still inside the domain
				if not self.isinside(x):
					constraints = [{'type':'eq',
									'fun': lambda z:np.dot(U_norm.T, self.normalize(z)) - np.dot(U_norm.T, x_norm),
									}]
					if self.b.shape[0] > 0:
						constraints.append({'type': 'ineq',
									 'fun': lambda z: self.b - np.dot(self.A, z),
									 'jac': lambda z: -self.A})
					
					res = minimize(lambda z: np.linalg.norm(x-z),
							x,
							bounds = [(lb_, ub_) for lb_,ub_ in zip(self.lb, self.ub)],
							constraints = constraints,
						  )
					x = res.x
					
				if self.isinside(x):
					X.append(x)
					it += 1

		return np.array(X)	

	def chebyshev_center(self):
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
			A.append(ei)
			b.append(self.ub[i])
			# Lower bound
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
		if hasattr(self, '_radius'):	
			return self._radius
		else:
			center, radius = self.chebyshev_center()
			return self._radius

class ComboDomain(Domain):
	""" Holds multiple domains together
	"""
	def __init__(self, domains = None):
		if domains == None:
			domains = []
		self.domains = deepcopy(domains)

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

	def _split(self, X):
		original_dim = len(X.shape)
		start = 0
		stop = 0
		X_vec = []
		X = np.atleast_2d(X)	
		for dom in self.domains:
			stop += len(dom)
			#print start, stop
			X_vec.append(X[:,start:stop])
			start = stop
		if original_dim == 1:
			X_vec = [x.flatten() for x in X_vec]
		return X_vec


	def sample(self, draw = 1):
		x = []
		for dom in self.domains:
			x.append(dom.sample(draw = draw))
		return np.hstack(x)

	def isinside(self, x, verbose = False):
		state = [dom.isinside(x_vec) for x_vec, dom in zip(self._split(x), self.domains)]
		if verbose:
			print state
		return all(state)

	def extent(self, x, p):
		alpha = [dom.extent(xx, pp) for dom, xx, pp in zip(self.domains, self._split(x), self._split(p))]
		return min(alpha)

	def normalize(self, X):
		original_dim = len(X.shape)
		X = np.atleast_2d(X)
		X_norm = np.hstack([dom.normalize(x) for dom, x in zip(self.domains, self._split(X))])
		if original_dim == 1:
			X_norm = X_norm.flatten()
		return X_norm

	def unnormalize(self, X_norm):
		original_dim = len(X_norm.shape)
		X_norm = np.atleast_2d(X_norm)
		X = np.hstack([dom.unnormalize(x) for dom, x in zip(self.domains, self._split(X_norm))])
		if original_dim == 1:
			X = X.flatten()
		return X

	def __len__(self):
		return sum([len(dom) for dom in self.domains])


	@property
	def lb(self):
		return np.hstack([dom.lb for dom in self.domains])
	
	@property
	def lb_norm(self):
		return np.hstack([dom.lb_norm for dom in self.domains])
		
	
	@property
	def ub(self):
		return np.hstack([dom.ub for dom in self.domains])
	
	@property
	def ub_norm(self):
		return np.hstack([dom.ub_norm for dom in self.domains])


	@property
	def A(self):
		As = [dom.A for dom in self.domains]
		A = np.zeros((sum([A_.shape[0] for A_ in As]), self.__len__() ))
		row_start = 0
		col_start = 0
		for A_ in As:
			A[row_start:row_start+A_.shape[0], col_start:col_start + A_.shape[1]] = A_
			row_start += A_.shape[0]
			col_start += A_.shape[1]
	
		return A
	
	@property
	def A_norm(self):
		As = [dom.A_norm for dom in self.domains]
		A = np.zeros((sum([A_.shape[0] for A_ in As]), self.__len__() ))
		row_start = 0
		col_start = 0
		for A_ in As:
			A[row_start:row_start+A_.shape[0], col_start:col_start + A_.shape[1]] = A_
			row_start += A_.shape[0]
			col_start += A_.shape[1]
	
		return A
	
	@property
	def b(self):
		return np.hstack([dom.b for dom in self.domains])
	@property
	def b_norm(self):
		return np.hstack([dom.b_norm for dom in self.domains])
	

	@property
	def A_eq(self):
		return np.hstack([dom.A_eq for dom in self.domains])
	@property
	def A_eq_norm(self):
		return np.hstack([dom.A_eq_norm for dom in self.domains])
	
	@property
	def b_eq(self):
		return np.hstack([dom.b_eq for dom in self.domains])
	@property
	def b_eq_norm(self):
		return np.hstack([dom.b_eq_norm for dom in self.domains])

	def dist(self, x, y):
		return sum([dom.dist(x_, y_) for dom, x_, y_ in zip(self.domains, self._split(x), self._split(y)) ])

	@property
	def center(self):
		return np.hstack([dom.center for dom in self.domains])

	def normalized_domain(self):
		domains = [dom.normalized_domain() for dom in self.domains]	
		return ComboDomain(domains)

class BoxDomain(Domain):
	
	def __init__(self, lb, ub, center = None):
		"""Uniform Sampling on a Box
		repeat : if scalar domain, 
		"""
		lb = np.array(lb).reshape(-1)
		ub = np.array(ub).reshape(-1)
		assert lb.shape[0] == ub.shape[0], "lower and upper bounds must have the same length"
		assert np.all(lb < ub)
		# TODO: Make it so that ordering can be switched
		self.lb = lb 
		self.ub = ub
		if center is None:
			center = (ub + lb)/2.
		self.center = center
		self.A = np.zeros((0,self.__len__()))
		self.b = np.zeros((0,))

		self.A_eq = np.zeros((0,self.__len__()))
		self.b_eq = np.zeros((0,))


		# Normalized doamin
		self.lb_norm = -np.ones(lb.shape)
		self.ub_norm = np.ones(lb.shape)
		self.A_norm = self.A
		self.b_norm = self.b
		
		self.A_eq_norm = self.A_eq
		self.b_eq_norm = self.b_eq

	def __len__(self):
		return self.lb.shape[0]

	#@doc_inherit	
	def sample(self, draw = 1):

		x_sample = np.random.uniform(self.lb, self.ub, size = (draw, self.lb.shape[0]))
		#if x is not None:
		#	# Replace those points 
		#	I = np.isnan(x)
		#	x_sample[I] = x[I]
		if draw == 1:
			return x_sample.flatten()
		return x_sample
		#return np.repeat(x_sample, self.repeat)

	#@doc_inherit	
	def isinside(self, x):
		if len(x.shape) == 1:
			#if self.repeat > 1:
			#	if np.any(x[0] != x):
			#		return False
			if np.any(x < self.lb) or np.any(x > self.ub):
				return False
			return True
		elif len(x.shape) == 2:
			raise NotImplementedError

	#@doc_inherit	
	def normalize(self, X):
		"""
		Return points shifted and scaled to [-1,1]^m.
		"""
		X = np.atleast_1d(np.array(X))
		original_dim = len(X.shape)
		X = np.atleast_2d(X)
	
	
		lb = self.lb.reshape(1, -1)
		ub = self.ub.reshape(1, -1)
		# Idenify parameters with zero range
		I = self.ub != self.lb
		norm = np.zeros(X.shape)
		# Normalize parameters with non-zero range
		norm[:,I] = 2.0 * (X[:,I] - lb[:,I]) / (ub[:,I] - lb[:,I]) - 1.0
		# Simply offset the rest
		norm[:,~I] = X[:,~I] - lb[:,~I]	

		if original_dim == 1:
			return norm.flatten()
		else:
			return norm

	#@doc_inherit	
	def unnormalize(self, X):
		"""
		Return points shifted and scaled to (lb, ub).
		"""
		original_dim = len(X.shape)
		X = np.atleast_2d(X)
		X.reshape(-1, self.lb.shape[0])
		lb = self.lb.reshape(1, -1)
		ub = self.ub.reshape(1, -1)
		app = (ub - lb) * (X + 1.0) / 2.0 + lb
		# If there is zero range, fix those
		I = (ub == lb).flatten()
		app[:,I] = lb[0,I]
		if original_dim == 1:
			return app.flatten()
		else:
			return app

	def extent(self, x, p):
		alpha = float('inf')
		# Now check box constraints
		I = np.nonzero(p)
		y = (self.ub - x)[I]/p[I]
		if np.sum(y>0) > 0:
			alpha = min(alpha, np.min(y[y>0]))	
	
		y = (self.lb - x)[I]/p[I]
		if np.sum(y>0) > 0:
			alpha = min(alpha, np.min(y[y>0]))

		# If on the boundary, the direction needs to point inside the domain
		if np.any(p[self.lb == x] < 0):
			alpha = 0
		if np.any(p[self.ub == x] > 0):
			alpha = 0	
		return alpha

	def normalized_domain(self):
		return BoxDomain( -np.ones(self.lb.shape), np.ones(self.ub.shape), self.normalize(self.center))
		

class UniformDomain(BoxDomain):
	pass

# TODO: The lower bounds/upper bounds are not a good constraint due to the way clipping is handled 
class NormalDomain(Domain):
	def __init__(self, mean, cov = None, clip = None):
		"""

		clip: float or None
			If not None, clip the values of the Gaussian distribution to be limited.
		"""
		if isinstance(mean, float) or isinstance(mean, int):
			mean = [mean]
		if isinstance(cov, float) or isinstance(mean, int):
			cov = [[cov]]
		self.mean = np.array(mean)
		m = self.mean.shape[0]
		if cov is None:
			cov = np.eye(m)
		self.cov = np.array(cov)
		self.ew, self.ev = np.linalg.eigh(cov)
		# np.dot(dom.ev, np.dot(np.diag(dom.ew), dom.ev.T)) = cov
		self.clip = clip
		assert np.all(self.ew > 0), 'covariance matrix must be positive definite'

	def isinside(self, x):
		if self.clip is None:
			if len(x.shape) == 1:
				return True		
			elif len(x.shape) == 2:
				return np.ones(x.shape[0], dtype = np.bool)
		else:
			x_norm = self.normalize(x)
			if len(x.shape) == 1:
				return np.linalg.norm(x_norm, 2) < self.clip
			else:
				xx = np.sqrt(np.sum(x_norm**2, axis = 1))
				return np.array([ x_ < self.clip for x_ in xx], dtype = np.bool)
				

	def extent(self, x, p):
		if self.clip is None:
			return float('inf')
		else:
			assert self.isinside(x), "Starting point is not inside the domain"
			def dist(alpha):
				return np.linalg.norm(self.normalize(x + alpha*p),2) - self.clip

			return auto_root(dist)		
	
	def normalize(self, x):
		if len(x.shape) == 1:
			return np.dot(self.ev, np.dot(np.diag(1./np.sqrt(self.ew)), x - self.mean))
		elif len(x.shape) == 2:
			return np.dot(self.ev, np.dot(np.diag(1./np.sqrt(self.ew)), x.T - self.mean.reshape(-1,1))).T
		raise NotImplementedError

	def unnormalize(self, y):
		if len(y.shape) == 1:
			return self.mean + np.dot(np.diag(np.sqrt(self.ew)), np.dot(self.ev.T, y))
		elif len(y.shape) == 2:
			return (self.mean.reshape(-1,1) + np.dot(np.diag(np.sqrt(self.ew)), np.dot(self.ev.T, y.T))).T
		raise NotImplementedError

	def sample(self, draw = 1):
		x = np.random.randn(draw, self.mean.shape[0])
		while self.clip is not None:
			# Remove samples violating clipping constraint
			xx = np.sqrt(np.sum(x**2, axis = 1))
			I = xx > self.clip
			if np.sum(I) > 0:
				x[I] = np.random.randn(np.sum(I), self.mean.shape[0])
			else:
				break

		x = self.unnormalize(x)
		#x = np.random.multivariate_normal(self.mean, self.cov, size = (draw, self.mean.shape[0]))
		if draw == 1:
			return x.flatten()
		else:
			return x.reshape(draw,-1)
		return x
	
	def __len__(self):
		return self.mean.shape[0]


	def dist(self, x, y):
		x = self.normalize(x)
		y = self.normalize(y)
		return np.linalg.norm(x-y)**2

	@property
	def lb(self):
		if self.clip is None:
			return -np.inf * np.ones(self.__len__())
		else:
			return self.mean.flatten() - self.clip*np.sqrt(np.diag(self.cov))

	@property
	def lb_norm(self):
		if self.clip is None:
			return -np.inf * np.ones(self.__len__())
		else:
			return -self.clip * np.ones(self.__len__())

	@property
	def ub(self):
		if self.clip is None:
			return np.inf * np.ones(self.__len__())
		else:
			return self.mean.flatten() + self.clip*np.sqrt(np.diag(self.cov))
	
	@property
	def ub_norm(self):
		if self.clip is None:
			return np.inf * np.ones(self.__len__())
		else:
			return self.clip * np.ones(self.__len__())
			

	@property
	def A(self):
		return np.zeros((0,self.__len__()))

	@property
	def A_norm(self):
		return np.zeros((0,self.__len__()))
	
	@property
	def A_eq(self):
		return np.zeros((0,self.__len__()))

	@property
	def A_eq_norm(self):
		return np.zeros((0,self.__len__()))

	@property
	def b(self):
		return np.zeros((0,))
	
	@property
	def b_norm(self):
		return np.zeros((0,))

	@property
	def b_eq(self):
		return np.zeros((0,))
	
	@property
	def b_eq_norm(self):
		return np.zeros((0,))

	@property
	def center(self):
		return self.mean

	@property
	def center_norm(self):
		return np.zeros((self.__len__()))


	def normalized_domain(self):
		return NormalDomain(np.zeros(self.mean.shape), clip = self.clip) 

class LogNormalDomain(Domain):
	def __init__(self, mean, cov = None, scaling = 1., clip = None):
		if isinstance(mean, float) or isinstance(mean, int):
			mean = [mean]
		if isinstance(cov, float) or isinstance(mean, int):
			cov = [[cov]]
		self.mean = np.array(mean)
		m = self.mean.shape[0]
		if cov is None:
			cov = np.eye(m)
	
		self.cov = np.array(cov)
		if self.mean.shape[0] == 1:
			self.cov.reshape(1,1)
		
		self.ew, self.ev = np.linalg.eigh(cov)
		self.scaling = scaling
		self.clip = clip
		assert np.all(self.ew > 0), 'covariance matrix must be positive definite'


	def isinside(self, x):
		if self.clip is None:
			if len(x.shape) == 1:
				return np.all(x > 0)
			else:
				return np.array([np.all(xx > 0) for xx in x], dtype = np.bool)
		else:
			x_norm = self.normalize(x)
			if len(x.shape) == 1:
				return np.linalg.norm(x_norm, 2) < self.clip
			else:
				xx = np.sqrt(np.sum(x_norm**2, axis = 1))
				return np.array([ x_ < self.clip for x_ in xx], dtype = np.bool)

	def extent(self, x, p):
		assert self.isinside(x), "Starting point is not inside the domain"
		# Check that the value doesn't go negative
		alpha = float('inf')
		if np.sum(p < 0) > 0:
			alpha = min(alpha, np.nanmin(-x[p<0]/p[p<0]))
		
		if self.clip is not None:
			def dist(gamma):
				if np.min(x + gamma*p)< 0:
					return np.inf
				return np.linalg.norm(self.normalize(x + gamma*p),2) - self.clip

			alpha_brent = auto_root(dist)		
			#print x, p, alpha, alpha_brent, dist(alpha_brent)
			alpha = min(alpha, alpha_brent)
			
		return alpha
	
	def normalize(self, x):
		if len(x.shape) == 1:
			return np.dot(self.ev, np.dot(np.diag(1./np.sqrt(self.ew)), np.log(x/self.scaling) - self.mean))
		elif len(x.shape) == 2:
			return np.dot(self.ev, np.dot(np.diag(1./np.sqrt(self.ew)), np.log(x/self.scaling).T - self.mean.reshape(-1,1))).T
		raise NotImplementedError

	def unnormalize(self, y):
		if len(y.shape) == 1:
			return np.exp(self.mean + np.dot(np.diag(np.sqrt(self.ew)), np.dot(self.ev.T, y)))*self.scaling
		elif len(y.shape) == 2:
			return np.exp(self.mean.reshape(-1,1) + np.dot(np.diag(np.sqrt(self.ew)), np.dot(self.ev.T, y.T))).T*self.scaling
		raise NotImplementedError

	def sample(self, draw = 1):
		x = np.random.randn(draw, self.mean.shape[0])
		while self.clip is not None:
			# Remove samples violating clipping constraint
			xx = np.sqrt(np.sum(x**2, axis = 1))
			I = xx > self.clip
			if np.sum(I) > 0:
				x[I] = np.random.randn(np.sum(I), self.mean.shape[0])
			else:
				break
		x = self.unnormalize(x)
		#x = np.exp(np.random.multivariate_normal(self.mean, self.cov, size = (draw, self.mean.shape[0])))*self.scaling
		if draw == 1:
			return x.reshape(self.mean.shape[0])
		return x.reshape(draw, self.mean.shape[0])

	def __len__(self):
		return self.mean.shape[0]


	@property
	def lb(self):
		if self.clip is None:
			return np.zeros(self.__len__())		
		else:
			return np.exp(self.mean - self.clip*np.sqrt(np.diag(self.cov)))*self.scaling

	@property
	def lb_norm(self):
		if self.clip is None:
			return -np.inf * np.ones(self.__len__())		
		else:
			return -self.clip * np.ones(self.__len__())

	@property
	def ub(self):
		if self.clip is None:
			return np.inf * np.ones(self.__len__())
		else:
			return np.exp(self.mean + self.clip*np.sqrt(np.diag(self.cov)))*self.scaling
	
	@property
	def ub_norm(self):
		if self.clip is None:
			return np.inf * np.ones(self.__len__())		
		else:
			return self.clip * np.ones(self.__len__())

	@property
	def A(self):
		return np.zeros((0,self.__len__()))
		
	@property
	def A_norm(self):
		return np.zeros((0,self.__len__()))
	
	@property
	def A_eq(self):
		return np.zeros((0,self.__len__()))
		
	@property
	def A_eq_norm(self):
		return np.zeros((0,self.__len__()))

	@property
	def b(self):
		return np.zeros((0,))
	
	@property
	def b_norm(self):
		return np.zeros((0,))
	
	@property
	def b_eq(self):
		return np.zeros((0,))
	
	@property
	def b_eq_norm(self):
		return np.zeros((0,))
	
	def dist(self, x, y):
		x = self.normalize(x)
		y = self.normalize(y)
		return np.linalg.norm(x - y)**2

	@property
	def center(self):
		return self.scaling*np.exp(self.mean)

	@property
	def center_norm(self):
		return np.zeros(self.mean.shape)
	
	def normalized_domain(self):
		return NormalDomain(np.zeros(self.mean.shape), clip = self.clip) 

class LinIneqDomain(BoxDomain, Domain):
	"""Defines a domain specified by a linear inequality constraint

	This defines a domain 

		{x: A x <= b and lb <= x <= ub and A_eq x = b_eq}

	"""
	
	def __init__(self, A, b, lb = None, ub = None, center = None, A_eq = None, b_eq = None):
		"""

			Warning: equality constraints are currently only supported by the random sampler
		"""
		m, n = A.shape

		self.A = np.copy(A)
		self.b = np.copy(b)

		if A_eq is not None:
			self.A_eq = np.copy(A_eq)
			self.A_eq_basis = orth(A_eq.T)
		else:
			self.A_eq = np.zeros((0,self.A.shape[1]))
			self.A_eq_basis = np.zeros((0,self.A.shape[1])).T 

		if b_eq is not None:
			self.b_eq = np.copy(b_eq)
		else:
			self.b_eq = np.zeros((0,))

		if lb is not None:
			rows = np.isfinite(lb)
			self.lb = np.copy(lb)
			self.lb[~rows] = -np.inf
		else:
			self.lb = -np.inf * np.ones(n)

		if ub is not None:
			rows = np.isfinite(ub)
			self.ub = np.copy(ub)
			self.ub[~rows] = np.inf
		else:
			self.ub = np.inf * np.ones(n)

		m, n = self.A.shape
		
		assert n == self.lb.shape[0]
		assert n == self.ub.shape[0]


		# Correct infinite domains
		for i in range(n):
			if not np.isfinite(self.lb[i]):
				c = np.zeros((n,))
				c[i] = 1
				x = linprog(c, A_ub = self.A, b_ub = self.b, A_eq = self.A_eq, b_eq = self.b_eq)
				self.lb[i] = x[i]
	
			if not np.isfinite(self.ub[i]):
				c = np.zeros((n,))
				c[i] = -1
				x = linprog(c, A_ub = self.A, b_ub = self.b, A_eq = self.A_eq, b_eq = self.b_eq)
				self.ub[i] = x[i]



		self.only_bounds = np.array([np.linalg.norm(self.A[:,i]) < 1e-10 and \
							np.isfinite(self.lb[i]) and np.isfinite(self.ub[i]) for i in range(n)])
		
		if self.A_eq.shape[0] > 0:
			self.only_bounds *= np.array([np.linalg.norm(self.A_eq[:,i])< 1e-10 for i in range(n)])
		
		if center is None:
			# get an initial feasible point using the Chebyshev center. 
			center, radius = self.chebyshev_center() 
			


		# Check we are a valid point
		assert np.all(self.lb - 1e-7 <= center), "failed LB test by %g" % np.max(center - self.lb)
		assert np.all(self.ub + 1e-7 >= center), "failed UB test by %g" % np.max(self.ub - center)
		assert np.all(np.dot(self.A, center) <= self.b + 1e-7), "failed inequality test by %g" % np.max(self.b - np.dot(self.A, center))
		
		if self.A_eq is not None:
			# Check that our center satisfies the equality constraints
			assert np.all(np.abs(np.dot(self.A_eq, center)-self.b_eq)< 1e-7), "failed equality test by %g" % np.max(np.abs(np.dot(self.A_eq, center) - self.b_eq))

		self.center = center	
		self.z0 = np.copy(self.center)

	
		# Bind the normalization functions from BoxDomain
		self.normalize = super(LinIneqDomain, self).normalize
		self.unnormalize = super(LinIneqDomain, self).unnormalize



	def __len__(self):
		return self.center.shape[0]
			
	def isinside(self, x):
		if len(x.shape) == 1:
			#print np.all(np.dot(self.A, x) <= self.b), np.all(self.lb <= x), np.all(x <= self.ub)
			return np.all(np.dot(self.A, x) <= self.b + 1e-7) and np.all(self.lb - 1e-7 <= x) and np.all(x <= self.ub + 1e-7)	


	def extent(self, x, p):
		alpha = np.inf
		# positive extent
		y = (self.b - np.dot(self.A, x)	)/np.dot(self.A, p)
		if np.sum(y>0) > 0:
			alpha = min(alpha, np.min(y[y>0]))
		# Now check box constraints
		I = np.nonzero(p)
		y = (self.ub - x)[I]/p[I]
		if np.sum(y>0) > 0:
			alpha = min(alpha, np.min(y[y>0]))	
	
		y = (self.lb - x)[I]/p[I]
		if np.sum(y>0) > 0:
			alpha = min(alpha, np.min(y[y>0]))


		# If on the boundary, the direction needs to point inside the domain
		if np.any(p[self.lb == x] < 0):
			alpha = 0
		if np.any(p[self.ub == x] > 0):
			alpha = 0	
		return alpha


	def _sample(self, no_recurse = False):
		maxiter = 5000
		bad_dir = True
		for it in range(maxiter):
			p = np.random.normal(size = self.center.shape)

			# Make sure the random step is orthogonal to the equality constraints
			if self.A_eq.shape[0] > 0:
				p -= np.dot(self.A_eq_basis, np.dot(self.A_eq_basis.T, p))
				#p -= np.dot(self.A_eq.T, np.dot(self.A_eq, p))
			
			# TODO: Why is this needed?	
			#p[self.only_bounds] = 0.
			if np.linalg.norm(p) < 1e-16:
				bad_dir = False
				break
			alpha_min = -self.extent(self.z0, -p)
			alpha_max = self.extent(self.z0, p)
			if alpha_max - alpha_min > 1e-10:
				bad_dir = False
				break

		if bad_dir and no_recurse ==  False:
			self.z0 = np.copy(self.center)
			return self._sample(no_recurse = True)

		if bad_dir == True and no_recurse == True:
			raise Exception('could not find a good direction')
		
		# update the center
		step_length = np.random.uniform(alpha_min, alpha_max)
		
		# update the current location
		self.z0 += step_length * p
	
		# Sample randomly on the variables that only have bound constraints	
		z = np.copy(self.z0)
		z[self.only_bounds] = np.random.uniform(self.lb[self.only_bounds], self.ub[self.only_bounds])	
		return z
	
	def sample(self, draw = 1, burn = 3):
		if draw == 1:
			for i in range(burn):
				self._sample()
			return self._sample()
		else:
			X = []
			for j in range(draw):
				for i in range(burn):
					self._sample()
				X.append(self._sample())
			return np.array(X)


	@property
	def A_norm(self):
		D = np.diag( (self.ub - self.lb)/2.0)
		return np.dot(self.A, D)
	
	@property
	def A_eq_norm(self):
		D = np.diag( (self.ub - self.lb)/2.0)
		return np.dot(self.A_eq, D)
	
	@property
	def b_norm(self):
		c = (self.ub + self.lb)/2.0
		return self.b - np.dot(self.A, c)

	@property
	def b_eq_norm(self):
		c = (self.ub + self.lb)/2.0
		return self.b_eq - np.dot(self.A_eq, c)
	
	@property
	def lb_norm(self):
		return -np.ones(len(self))
	
	@property
	def ub_norm(self):
		return np.ones(len(self))
	
	def normalized_domain(self):
		return LinIneqDomain(self.A_norm, self.b_norm, lb = self.lb_norm, ub = self.ub_norm)



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
		else:
			lb = np.atleast_1d(np.min(X))
			ub = np.atleast_1d(np.max(X))
			A = np.zeros((0, 1))
			b = np.zeros((0,))
			LinIneqDomain.__init__(self, A, b, lb = lb, ub = ub)
	

