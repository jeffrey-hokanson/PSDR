
class Function(object):
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

	def __get__(self, i):
		"""Get a particular sub-function as another Function"""
		raise NotImplemented

class WrapperFunction(Function):
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
	def __init__(self, funs, domain, vectorized = False, fun_kwargs = None):
		self.funs = funs
		self.domain = domain

	
