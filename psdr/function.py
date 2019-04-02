import numpy as np
import dill
# We set this so that functions can be transparently pickled
# from the users perspective
dill.settings['recurse'] = True

from domains import Domain


__all__ = ['Function', 'BaseFunction']


class BaseFunction(object):
	r""" Abstract base class for functions

	"""
	def eval(self, X):
		return self.__call__(X, return_grad = False)

	def grad(self, X):
		return self.__call__(X, return_grad = True)[1]

	def hessian(self, X):
		raise NotImplementedError

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
	dask_client: dask.distributed.Client
		Client to use for multiprocessing 
	"""

	# TODO: Implement fancy pool-features

	def __init__(self, funs, domain, grads = None, fd_grad = None, vectorized = False, fun_kwargs = None,
		dask_client = None):

		self.dask_client = dask_client

		self.vectorized = vectorized
		
		if callable(funs):
			self._funs = [funs]
		else:
			self._funs = funs

		if grads is not None:
			if callable(grads):
				grads = [grads]
			assert len(grads) == len(self._funs), "Must provide the same number of functions and gradients"
			self._grads = grads

		if dask_client is not None:
			# Pickle the functions for later use when calling distributed code
			self._funs_dill = []
			for fun in self._funs:	
				# A big problem is when functions are imported from another module
				# dill will simply want to import these functions;
				# i.e., the function is stored by a referrence to the file in which it originated.
				# See discussion at https://github.com/uqfoundation/dill/issues/123
				
				# There are also otherways to handle this problem.  For example, 
				# dask.distributed allows you to ship python files to the workers
				# see: https://stackoverflow.com/a/39295372/
			
				# So we do something more sophisticated in order to pickle these functions.
			
				# (1) We bring the function into the local scope, evaluating the function definition
				# inside this loop. 
				exec(dill.source.getsource(fun))

				# (2) We now pickle this function 
				# eval(fun.__name__) is the name of the function we just executed
				self._funs_dill.append(dill.dumps(eval(fun.__name__)))

				#self._funs_dill.append((dill.source.getsource(fun), fun.__name__ ))

		
		self.domain_app = domain
		self.domain_norm = domain.normalized_domain()
		self.domain = self.domain_norm
		self.fd_grad = fd_grad



	def eval(self, X_norm):
		X_norm = np.atleast_1d(X_norm)
		X = self.domain_app.unnormalize(X_norm)

		if len(X.shape) == 1:
			x = X.flatten()
			if callable(self._funs):
				return self._funs(x)
			else:
				return np.hstack([fun(x) for fun in self._funs])

		elif len(X.shape) == 2:
			if callable(self._funs):
				if self.vectorized:
					return self._funs(X)
				else:
					return np.vstack([self._funs(x) for x in X])
			else:
				if self.vectorized:
					return np.hstack([ np.array(fun(X)).reshape(-1,1) for fun in self._funs])
				else:
					return np.vstack([ np.hstack([fun(x) for fun in self._funs]) for x in X])
					
		else:
			raise NotImplementedError


	def eval_async(self, X_norm):
		r""" Evaluate the function asyncronously using dask.distributed
		"""
		assert self.dask_client is not None, "A dask_client must be specified on class initialization"

		X_norm = np.atleast_1d(X_norm)
		X = self.domain_app.unnormalize(X_norm)

		def subcall(funs_dill, x):
			import dill
			funs = [dill.loads(fun) for fun in funs_dill]
			return [fun(x) for fun in funs]

		return [self.dask_client.submit(subcall, self._funs_dill, x) for x in X]	
		

	def grad(self, X_norm):
		X_norm = np.atleast_1d(X_norm)

		# If we've asked to use a finite difference gradient
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

		X = self.domain_app.unnormalize(X_norm)
		D = self.domain_app._unnormalize_der() 	
		
		# Return gradient if specified
		if self._grads is not None: 
			if len(X.shape) == 1:
				if callable(self._grads):
					grad = self._grads(X)
				else:
					grad = np.vstack([grad(X) for grad in self._grads])
			
			elif len(X.shape) == 2:
				if callable(self._grads):
					if self.vectorized:
						grad = self._grads(X)
					else:
						grad = np.hstack([self._grads(x) for x in X])
				else: 
					if self.vectorized:
						grad = np.hstack([ np.array(grad(X)) for grad in self._grads])
					else:
						grad = np.hstack([ np.hstack([grad(x) for grad in self._grads]) for x in X])
			grad = D.dot(grad.T).T
			
			return grad

		# Try return_grad the function definition
		

	def __call__(self, X_norm, return_grad = False):
		if not return_grad:
			return self.eval(X_norm)

		if return_grad:
			try: 
				# TODO: Implement support for calling function with return_grad
				raise TypeError	

			except TypeError:
				return self.eval(X_norm), self.grad(X_norm)					
	
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

if __name__ == '__main__':
	from dask.distributed import Client
	from demos import borehole, build_borehole_domain, Borehole
	#from demos import Borehole


	#print(dill.source.getsource(borehole))
	#print(locals())
	#exec(dill.source.getsource(borehole))	
	#print(dill.dumps(borehole))

	client = Client('tcp://10.101.89.84:8786')

	fun = Borehole(dask_client = client)

	np.random.seed(0)	
	X = fun.domain.sample(10)
	res = fun.eval_async(X)
	for r in res:
		print(r.result())	


	#print(fun._funs_dill)
	#x = fun.domain_app.sample()
	#res = client.submit(borehole, x) 
	#print(res.result())	
	#print(fun._funs_dill)

	#dom = build_borehole_domain()
	#fun = Function(borehole, dom, dask_client = client)
	#print dill.dumps(borehole)
	#X = fun.domain.sample(10)

	#print fun.eval_async(X)


