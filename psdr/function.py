import numpy as np

from domains import Domain


__all__ = ['Function', 'BaseFunction']


class BaseFunction(object):
	r""" Abstract base class for functions

	"""
	def eval(self, X):
		return self.__call__(self, X, return_grad = False)

	def grad(self, X):
		return self.__call__(self, X, return_grad = True)[1]

	def __call__(self, X, return_grad = False):
		if return_grad:
			return self.eval(X), self.grad(X)
		else:
			return self.eval(X)


class Function(BaseFunction, Domain):
	r"""Wrapper around function specifying the domain

	Provided a function :math:`f: \mathcal{D} \subset \mathbb{R}^m \to \mathbb{R}^d`,
	and a domain :math:`\mathcal{D}`, this class acts as a wrapper for both.
	The key contribution of this class is to provide access to 
	the function on the *normalized domain* :math:`\mathcal{D}_{\text{norm}}` 
	that is a subset of the :math:`[-1,1]^m` cube; i.e.,

	.. math::

		\mathcal{D}_{\text{norm}} \subset [-1,1]^m \subset \mathbb{R}^m.

	Parameters
	----------
	fun: function or list of functions
		Either a python function or a list of functions to evaluate
	domain: Domain
		The domain on which the function is posed	
	vectorized: bool, default: False
		If True, the functions are vectorized for use with numpy.
	fun_kwargs: dict, default: empty
		Keyword arguments to pass to the functions when evaluating function 
	"""
	def __init__(self, funs, domain, grads = None, fd_grad = None, vectorized = False, fun_kwargs = None):
		self.funs = funs
		self.domain = domain
		self.domain_norm = domain.normalized_domain()
		self.fd_grad = fd_grad


	def eval(self, X_norm):
		X_norm = np.atleast_1d(X_norm)
		X = self.domain.unnormalize(X_norm)

		if len(X.shape) == 1:
			x = X.flatten()
			if callable(self.funs):
				return self.funs(x)
			else:
				return np.hstack([fun(x) for fun in self.funs])

		elif len(X.shape) == 2:
			if callable(self.funs):
				return np.vstack([self.funs(x) for x in X])
			else:
				return np.vstack([ np.hstack([fun(x) for fun in self.funs]) for x in X])
					
		else:
			raise NotImplementedError

	def grad(self, X_norm):
		X_norm = np.atleast_1d(X_norm)

		if self.fd_grad:
			h = 1e-7
			grads = []
			for x in X_norm:
				fx = self.eval(x)
				grad = np.zeros(x.shape)
				for i in range(len(x)):
					ei = np.zeros(x.shape)
					ei[i] = 1.
					grad[i] = (self.eval(x + h*ei) - fx)/h
				grads.append(grad)

			if len(X_norm.shape) == 1:
				return np.array(grads).flatten()
			else:
				return np.array(grads)

		else:
			raise NotImplementedError			
	
	def __get__(self, i):
		"""Get a particular sub-function as another Function"""
		raise NotImplemented

	##############################################################################
	# Domain Aliases
	##############################################################################

	def _closest_point(self, x0, L = None, **kwargs):
		return self.domain_norm._closest_point(x0, L = L, **kwargs)

	def _corner(self, p, **kwargs):
		return self.domain_norm._corner(p, **kwargs)

	def _extent(self, x, p):
		return self.domain_norm._extent(x, p)

	def _isinside(self, X):
		return self.domain_norm._isinside(X)	 
	
	def _constrained_least_squares(self, A, b, **kwargs):
		return self.domain_norm._constrained_least_squares(A, b, **kwargs)

	def _sample(self, draw = 1):
		return self.domain_norm._sample(draw)


	# We remove multiplication because it doesn't make any sense with how it interacts with
	# the call statements for the function
	def __mul__(self, other):
		raise NotImplementedError 
	
	def __rmul__(self, other):
		raise NotImplementedError 


