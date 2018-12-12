# Subspace based dimension reduction techniques
from __future__ import division
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import cvxpy as cp


__all__ = ['SubspaceBasedDimensionReduction',
	'ActiveSubspace', 
	'LipschitzMatrix',
	]

class SubspaceBasedDimensionReduction(object):
	r""" Abstract base class for Subspace-Based Dimension Reduction

	Given a function :math:`f : \mathcal{D} \to \mathbb{R}`, 
	subspace-based dimension reduction identifies a subspace, 
	described by a matrix :math:`\mathbf{U} \in \mathbb{R}^{n\times m}`
	with orthonormal columns for some :math:`n \le m`.

	"""
	@property
	def U(self):
		""" A matrix defining the 'important' directions

		Returns
		-------
		np.ndarray (n,m):
			Matrix with orthonormal columns defining the important directions in decreasing order
			of precidence.
		"""
		raise NotImplementedError


	def shadow_plot(self, X = None, fX = None, dim = 1, ax = None):
		r""" Draw a shadow plot


		Parameters
		----------
		X: array-like (N,m)
			Input coordinates for function samples
		fX: array-like (N,)
			Values of function at sample points
		dim: int, [1,2]
			Dimension of shadow plot
		ax: matplotlib.pyplot.axis
			Axis on which to draw the shadow plot
		"""
		if ax is None:
			fig, ax = plt.subplots()
		
		if dim == 1:
			ax.plot(X.dot(self.U[:,0]), fX, 'k.')
			ax.set_xlabel(r'active coordinate $\mathbf{u}^\top \mathbf{x}$')
			ax.set_ylabel(r'$f(\mathbf{x})$')
		else:
			raise NotImplementedError		

		return ax

	def _init_dim(self, X = None, grads = None):
		if X is not None:
			self._dimension = len(X[0])
		elif grads is not None:
			self._dimension = len(grads[0])
		else:
			raise Exception("Could not determine dimension of ambient space")


	def __len__(self):
		return self._dimension

	@property
	def X(self):
		return np.zeros((0,len(self)))
	
	@property
	def fX(self):
		return np.zeros((0,len(self)))

	@property
	def grads(self):
		return np.zeros((0,len(self)))


class ActiveSubspace(SubspaceBasedDimensionReduction):
	r"""Computes the active subspace based on gradient samples

	Given the function :math:`f:\mathcal{D} \to \mathbb{R}`,
	the active subspace is defined as the eigenvectors corresponding to the 
	largest eigenvalues of the average outer-product of gradients:

	.. math::

		\mathbf{C} := \int_{\mathbf{x}\in \mathcal{D}} \nabla f(\mathbf{x}) \nabla f(\mathbf{x})^\top \  \mathrm{d}\mathbf{x}
		\in \mathbb{R}^{m\times m}.

	By default, this class assumes that we are provided with gradients
	evaluated at random samples over the domain and estimates the matrix :math:`\mathbf{C}`
	using Monte-Carlo integration. However, if provided a weight corresponding to a quadrature rule,
	this will be used instead to approximate this matrix; i.e.,
		
	.. math::

		\mathbf{C} \approx \sum_{i=1}^N w_i \nabla f(\mathbf{x}_i) \nabla f(\mathbf{x}_i)^\top.

	
	Parameters
	----------
	grads: array-like (N,m)
		Gradient samples of function (tacitly assumed to be uniform on the domain
		or from a quadrature rule with corresponding weight).
	weights: array-like (N,), optional
		Weights corresponding to a quadrature rule associated with the samples of the gradient.

	"""
	def __init__(self, grads, weights = None):
		self._init_dim(grads = grads)

		self._grads = np.array(grads).reshape(-1,len(self))
		N = len(self._grads)
		if weights is None:
			weights = np.ones(N)/N
			
		self._weights = np.array(weights)
		self._U, self._s, VT = np.linalg.svd(np.sqrt(self._weights)*self._grads.T)


	@property
	def U(self):
		return np.copy(self._U)


class LipschitzMatrix(SubspaceBasedDimensionReduction):
	r"""Constructs the subspace-based dimension reduction from the Lipschitz Matrix.

	The Lipschitz matrix :math:`\mathbf{L} \in \mathbb{R}^{m \times m}` a matrix that 
	acts analogously to the Lipschitz constant, defining a function class where

	.. math::

		\lbrace f: \mathbb{R}^m\to \mathbb{R}: |f(\mathbf{x}_1) - f(\mathbf{x}_2)| \le \|\mathbf{L}(\mathbf{x}_1 - \mathbf{x}_2\|_2 \rbrace.

	In general we cannot determine the Lipschitz matrix analytically. 
	Instead we seek to estimate it via the lower bound based on samples :math:`\lbrace \mathbf{x}_i, f(\mathbf{x}_i)\rbrace_i`
	and/or gradients :math:`\lbrace \nabla f(\mathbf{x}_i)\rbrace_i`.
	Here we do so by solving a semidefinite program for the symmetric positive definite matrix :math:`\mathbf{M} \in \mathbb{R}^{m\times m}`:

	.. math::

		\min_{\mathbf{M} \in \mathbb{S}^{m\times m}} & \ \text{Trace } \mathbf{M} \\
		\text{such that} & \ |f(\mathbf{x}_i) - f(\mathbf{x}_j)|^2 \le (\mathbf{x}_i - \mathbf{x}_j)^\top \mathbf{M} (\mathbf{x}_i - \mathbf{x}_j) \\
		& \ \nabla f(\mathbf{x}_k) \nabla f(\mathbf{x}_k)^\top \preceq \mathbf{M}


	Parameters
	----------
	X : array-like (N, m), optional
		Input coordinates for function samples 

	"""
	def __init__(self, X = None, fX = None, grads = None, **kwargs):
		self._init_dim(X = X, grads = grads)

		if X is not None and fX is not None:
			N = len(X)
			assert len(fX) == N, "Dimension of input and output does not match"
			self._X = np.array(X).reshape(-1,m)
			self._fX = np.array(fX).reshape(len(self._X))
		elif X is None and fX is None:
			self._X = np.zeros((0,len(self)))
			self._fX = np.zeros((0,))
		else:
			raise AssertionError("X and fX must both be specified simultaneously or not specified")

		if grads is not None:
			self._grads = np.array(grads).reshape(-1,len(self))

		self._build_lipschitz_matrix(**kwargs)

		# Compute the important directions
		self._U, _, _ = np.linalg.svd(self._M)

		# Compute the Lipschitz matrix (lower triangular)
		self._L = scipy.linalg.cholesky(self.M[::-1][:,::-1], lower = False)[::-1][:,::-1]

	@property
	def X(self): return self._X
	
	@property
	def fX(self): return self._fX

	@property
	def grads(self): return self._grads

	@property
	def U(self): return np.copy(self._U)

	@property
	def M(self): 
		r""" The symmetric positive definite solution to the semidefinite program
		"""
		return self._M

	@property
	def L(self): 
		r""" The Lipschitz matrix estimate based on samples
		"""
		return self._L

	def _build_lipschitz_matrix(self, **kwargs):
		M = cp.Variable( (len(self), len(self)), PSD = True)
		
		# TODO: Implement normalization for function values/gradients for scaling purposes
		constraints = []
		
		# Sample constraint	
		for i in range(len(self.X)):
			for j in range(i+1, len(self.X)):
				lhs = (self.fX[i] - self.fX[j])**2
				y = self.X[i] - self.X[j]
				# y.T M y
				rhs = M.__matmul__(y).__rmatmul(y.T)
				constraints.append(lhs <= rhs)
			
		# gradient constraints
		for g in self.grads:
			constraints.append( np.outer(g,g) << M)

		problem = cp.Problem(cp.Minimize(cp.norm(M, 'fro')), constraints)
		problem.solve(**kwargs)
		
		self._M = np.array(M.value).reshape(len(self),len(self))
				


if __name__ == '__main__':
	X = np.random.randn(10,4)
	a = np.random.randn(4,)
	a = np.ones(4,)
	fX = np.dot(X, a).flatten()
	grads = np.tile(a, (X.shape[0], 1))
	lip = LipschitzMatrix(grads = grads)
	print lip.M
	print lip.L
	lip.shadow_plot(X = X, fX = fX)
	#act = ActiveSubspace(grads)
	#act.shadow_plot(X, fX)
	plt.show()
