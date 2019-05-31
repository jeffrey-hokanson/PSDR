from __future__ import print_function
import numpy as np
import textwrap
import inspect

import cloudpickle

#from .domains import Domain

from .misc import merge


__all__ = ['Function', 'BaseFunction']

# I've stopped using dill entirely because this issue prevents 
# me from loading modules inside functions
# https://github.com/uqfoundation/dill/issues/219	

class BaseFunction(object):
	r""" Abstract base class for functions

	"""
	def eval(self, X, **kwargs):
		return self.__call__(X, return_grad = False)

	def grad(self, X):
		return self.__call__(X, return_grad = True)[1]

	def hessian(self, X):
		raise NotImplementedError

	def __call__(self, X, return_grad = False, **kwargs):
		if return_grad:
			return self.eval(X, **kwargs), self.grad(X)
		else:
			return self.eval(X, **kwargs)


class Function(BaseFunction):
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
	kwargs: dict, default: empty
		Keyword arguments to pass to the functions when evaluating function
	dask_client: dask.distributed.Client
		Client to use for multiprocessing 
	"""

	def __init__(self, funs, domain, grads = None, fd_grad = None, vectorized = False, kwargs = {},
		dask_client = None, return_grad = False):

		self.dask_client = dask_client
		self.vectorized = vectorized
		self.kwargs = kwargs
		self.return_grad = return_grad
		
		if callable(funs):
			self._funs = [funs]
		else:
			self._funs = funs

		if grads is not None:
			if callable(grads):
				grads = [grads]
			assert len(grads) == len(self._funs), "Must provide the same number of functions and gradients"
			self._grads = grads
		else:
			self._grads = None

		if dask_client is not None:
			# Pickle the functions for later use when calling distributed code
			self._funs_pickle = []
			for fun in self._funs:	
				# A big problem is when functions are imported from another module
				# dill/cloudpickle will simply want to import these functions;
				# i.e., the function is stored by a referrence to the file in which it originated.
				# See discussion at https://github.com/uqfoundation/dill/issues/123
				
				# There are also otherways to handle this problem.  For example, 
				# dask.distributed allows you to ship python files to the workers
				# see: https://stackoverflow.com/a/39295372/
			
				# So we do something more sophisticated in order to pickle these functions.
			
				# (1) We bring the function into the local scope, evaluating the function definition
				# inside this loop.
				# Specifically, we run the code inside a custom scope in order to 
				# have the code load into dill/cloudpickle rather than passing around
				# as a reference.  The limited scope prevents this from overwriting any local functions

				# Get the code 
				code = inspect.getsource(fun)
				
				# Strip indentation 
				code = textwrap.dedent(code)
				
				# Execute code	
				scope = {}
				exec(code, scope, scope)
	
				# (2) We now pickle this function 
				# scope is a dictionary of functions, and the name allows us to specify which
				self._funs_pickle.append(cloudpickle.dumps(scope[fun.__name__]))

		self.domain_app = domain
		self.domain_norm = domain.normalized_domain()
		self.domain = self.domain_norm
		self.fd_grad = fd_grad

	def eval(self, X_norm, **kwargs):
		X_norm = np.atleast_1d(X_norm)
		X = self.domain_app.unnormalize(X_norm)

		kwargs = merge(self.kwargs, kwargs)

		if len(X.shape) == 1:
			x = X.flatten()
			return np.hstack([fun(x, **kwargs) for fun in self._funs]).flatten()

		elif len(X.shape) == 2:
			if self.vectorized:
				fX = [fun(X, **kwargs) for fun in self._funs]
				for fXi in fX:
					assert len(fXi) == X.shape[0], "Must provide an array with %d entires; got %d" % (X.shape[0], len(fXi) )
				
				# Reshape if necessary so concatention works
				for i, fXi in enumerate(fX):
					fXi = np.array(fXi)
					if len(fXi.shape) == 1:
						fX[i] = fXi.reshape(len(X),1)
				return np.hstack(fX)
			else:
				return np.vstack([ np.hstack([fun(x, **kwargs) for fun in self._funs]) for x in X])


	def eval_async(self, X_norm, **kwargs):
		r""" Evaluate the function asyncronously using dask.distributed
		"""
		assert self.dask_client is not None, "A dask_client must be specified on class initialization"
		
		kwargs = merge(self.kwargs, kwargs)

		X_norm = np.atleast_1d(X_norm)
		X = self.domain_app.unnormalize(X_norm)
		X = np.atleast_2d(X)

		def subcall(funs_pickle, x, **kwargs_):
			import cloudpickle
			funs = [cloudpickle.loads(fun) for fun in funs_pickle]
			return [fun(x, **kwargs_) for fun in funs]

		results = [self.dask_client.submit(subcall, self._funs_pickle, x, **kwargs) for x in X]	
		if len(X_norm.shape) == 1:
			return results[0]
		else:
			return results

	def grad(self, X_norm, **kwargs):
		kwargs = merge(self.kwargs, kwargs)
		
		X_norm = np.atleast_1d(X_norm)

		# If we've asked to use a finite difference gradient
		if self.fd_grad:
			h = 1e-7
			grads = []
			for x in np.atleast_2d(X_norm):
				fx = self.eval(x)
				grad = np.zeros(x.shape)
				for i in range(len(x)):
					ei = np.zeros(x.shape)
					ei[i] = 1.
					grad[i] = (self.eval(x + h*ei, **kwargs) - fx)/h
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
				grad = np.vstack([grad(X, **kwargs) for grad in self._grads])
			
			elif len(X.shape) == 2:
				if self.vectorized:
					grad = np.hstack([ np.array(grad(X, **kwargs)) for grad in self._grads])
				else:
					grad = np.vstack([ np.hstack([grad(x, **kwargs) for grad in self._grads]) for x in X])
			grad = D.dot(grad.T).T
			
			return grad

		# Try return_grad the function definition
		elif self.return_grad:
			if len(X.shape) == 1:
				grad = np.vstack([fun(X, return_grad = True, **kwargs)[1] for fun in self._funs])
				grads = grad.flatten()
				
			elif len(X.shape) == 2:
				if self.vectorized:
					grads = []
					for fun in self._funs:
						fXi, gradsi = fun(X, return_grad = True, **kwargs)
						grads.append(gradsi)
					grads = np.hstack([ np.array(grad) for grad in grads])
				else:
					grads = []
					for x in X:
						grad = []
						for fun in self._funs:
							fxi, gradi = fun(x, return_grad = True, **kwargs)
							grad.append(gradi)
						grads.append(np.hstack(grad))
					grads = np.vstack(grads)
			
			grads = D.dot(grads.T).T
			return grads
		else:
			raise NotImplementedError("Gradient not defined and finite-difference approximation not enabled")


	def __call__(self, X_norm, return_grad = False, **kwargs):
		kwargs = merge(self.kwargs, kwargs)
		if not return_grad:
			return self.eval(X_norm, **kwargs)

		if self.return_grad:
			X = self.domain_app.unnormalize(X_norm)
			X = np.atleast_2d(X)
			D = self.domain_app._unnormalize_der() 	
			# If the function can return both the value and gradient simultaneously
			if self.vectorized:
				ret = [fun(X, return_grad = True, **kwargs) for fun in self._funs]
				fX = np.hstack([r[0] for r in ret])
				grad = np.vstack([np.atleast_1d(r[1]) for r in ret])
			else:
				fX = []
				grad = []
				for x in X:
					fx = []
					g = []
					for fun in self._funs:
						fxi, gi = fun(x, return_grad = True, **kwargs)
						fx.append(fxi)
						g.append(gi)
					fX.append(np.hstack(fx))
					grad.append(np.hstack(g))

				fX = np.vstack(fX)
				grad = np.vstack(grad)

			grad = D.dot(grad.T).T
			if len(X_norm.shape) == 1:
				fX = fX.flatten()
				grad = grad.reshape(len(self.domain))
			return fX, grad
		else:
			return self.eval(X_norm, **kwargs), self.grad(X_norm, **kwargs)					

	def call_async(self, X_norm, return_grad = False, **kwargs):
		r""" Calls the function in an async. manner
		
		This mainly exists to cleanly separate eval_async which *only* returns function values
		and this function, call_async, which can optionally return gradients, like __call__.
		"""
		kwargs = merge(self.kwargs, kwargs)
		return self.eval_async(X_norm, return_grad = return_grad, **kwargs)
	
#	def __get__(self, i):
#		"""Get a particular sub-function as another Function"""
#		raise NotImplemented




