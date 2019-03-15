# Subspace based dimension reduction techniques
from __future__ import division, print_function
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import cvxpy as cp
import cvxopt

__all__ = ['SubspaceBasedDimensionReduction',
	'ActiveSubspace', 
	'LipschitzMatrix',
	]

class SubspaceBasedDimensionReduction(object):
	r""" Abstract base class for Subspace-Based Dimension Reduction

	Given a function :math:`f : \mathcal{D} \to \mathbb{R}`, 
	subspace-based dimension reduction identifies a subspace, 
	described by a matrix :math:`\mathbf{U} \in \mathbb{R}^{m\times n}`
	with orthonormal columns for some :math:`n \le m`.

	"""
	@property
	def U(self):
		""" A matrix defining the 'important' directions

		Returns
		-------
		np.ndarray (m, n):
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

		Returns
		-------
		ax: matplotlib.pyplot.axis
			Axis on which the plot is drawn
		"""
		if ax is None:
			if dim == 1:
				fig, ax = plt.subplots(figsize = (6,6))
			else:
				# Hack so that plot is approximately square after adding colorbar 
				fig, ax = plt.subplots(figsize = (7.5,6))
	
		if X is None:
			X = self.X
		
		if dim == 1:
			ax.plot(X.dot(self.U[:,0]), fX, 'k.')
			ax.set_xlabel(r'active coordinate $\mathbf{u}^\top \mathbf{x}$')
			ax.set_ylabel(r'$f(\mathbf{x})$')

		elif dim == 2:
			Y = self.U[:,0:2].T.dot(X.T).T
			sc = ax.scatter(Y[:,0], Y[:,1], c = fX.flatten(), s = 3)
			ax.set_xlabel(r'active coordinate 1 $\mathbf{u}_1^\top \mathbf{x}$')
			ax.set_ylabel(r'active coordinate 2 $\mathbf{u}_2^\top \mathbf{x}$')

			plt.colorbar(sc).set_label('f(x)')

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

	"""
	def __init__(self):
		self._U = None
		self._s = None


	def fit(self, grads, weights = None):
		r""" Find the active subspace

		Parameters
		----------
		grads: array-like (N,m)
			Gradient samples of function (tacitly assumed to be uniform on the domain
			or from a quadrature rule with corresponding weight).
		weights: array-like (N,), optional
			Weights corresponding to a quadrature rule associated with the samples of the gradient.

		"""
		self._init_dim(grads = grads)

		self._grads = np.array(grads).reshape(-1,len(self))
		N = len(self._grads)
		if weights is None:
			weights = np.ones(N)/N
			
		self._weights = np.array(weights)
		self._U, self._s, VT = np.linalg.svd(np.sqrt(self._weights)*self._grads.T)
	
		# TODO: Fix +/- scaling so average gradient is positive	

	@property
	def U(self):
		return np.copy(self._U)

	# TODO: Plot of eigenvalues (with optional boostrapped estimate)

	# TODO: Plot of eigenvector angles with bootstrapped replicates.


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

		\min_{\mathbf{H} \in \mathbb{S}^{m}_+} & \ \text{Trace } \mathbf{H} \\
		\text{such that} & \ |f(\mathbf{x}_i) - f(\mathbf{x}_j)|^2 \le (\mathbf{x}_i - \mathbf{x}_j)^\top \mathbf{H} (\mathbf{x}_i - \mathbf{x}_j) \\
		& \ \nabla f(\mathbf{x}_k) \nabla f(\mathbf{x}_k)^\top \preceq \mathbf{H}

	Parameters
	----------
	**kwargs: dict (optional)
		Additional parameters to pass to cvxpy
	"""
	def __init__(self, **kwargs):
		self._U = None
		self._L = None
		self.kwargs = kwargs
		if 'solver' not in self.kwargs:
			self.kwargs['solver'] = cp.CVXOPT

	def fit(self, X = None, fX = None, grads = None):
		r""" Find the Lipschitz matrix


		Parameters
		----------
		X : array-like (N, m), optional
			Input coordinates for function samples 
		fX: array-like (N,), optional
			Values of the function at X[i]
		grads: array-like (N,m), optional
			Gradients of the function evaluated anywhere	
		"""
		self._init_dim(X = X, grads = grads)

		if X is not None and fX is not None:
			N = len(X)
			assert len(fX) == N, "Dimension of input and output does not match"
			X = np.atleast_2d(np.array(X))
			self._dimension = X.shape[1]
			fX = np.array(fX).reshape(X.shape[0])

		elif X is None and fX is None:
			X = np.zeros((0,len(self)))
			fX = np.zeros((0,))
		else:
			raise AssertionError("X and fX must both be specified simultaneously or not specified")

		if grads is not None:
			grads = np.array(grads).reshape(-1,len(self))
		else:
			grads = np.zeros((0, len(self)))
		
		try:	
			scale1 = np.max(fX) - np.min(fX)
		except ValueError:
			scale1 = 1.
		try:
			scale2 = np.max([np.linalg.norm(grad) for grad in grads])
		except ValueError:
			scale2 = 1.
		scale = max(scale1, scale2)
		
		H = self._build_lipschitz_matrix_cvxopt(X, fX/scale, grads/scale)
		self._H = H = scale**2 * H

		# Compute the important directions
		#self._U, _, _ = np.linalg.svd(self._H)
		ew, U = scipy.linalg.eigh(self._H)
		# because eigenvalues are in ascending order, the subspace basis needs to be flipped
		self._U = U[:,::-1]

		# Compute the Lipschitz matrix (lower triangular)
		#self._L = scipy.linalg.cholesky(self.H[::-1][:,::-1], lower = False)[::-1][:,::-1]
		self._L = U.dot(np.diag(np.sqrt(np.maximum(ew, 0))).dot(U.T))

	@property
	def X(self): return self._X
	
	@property
	def fX(self): return self._fX

	@property
	def grads(self): return self._grads

	@property
	def U(self): return np.copy(self._U)

	@property
	def H(self): 
		r""" The symmetric positive definite solution to the semidefinite program
		"""
		return self._H

	@property
	def L(self): 
		r""" The Lipschitz matrix estimate based on samples
		"""
		return self._L

	def _build_lipschitz_matrix(self, X, fX, grads):
		H = cp.Variable( (len(self), len(self)), PSD = True)
		
		# TODO: Implement normalization for function values/gradients for scaling purposes
		constraints = []
		
		# Sample constraint	
		for i in range(len(X)):
			for j in range(i+1, len(X)):
				lhs = (fX[i] - fX[j])**2
				y = X[i] - X[j]
				# y.T M y
				#rhs = H.__matmul__(y).__rmatmul(y.T)
				rhs = cp.quad_form(y, H)
				constraints.append(lhs <= rhs)
			
		# gradient constraints
		for g in grads:
			constraints.append( np.outer(g,g) << H)

		problem = cp.Problem(cp.Minimize(cp.norm(H, 'fro')), constraints)
		problem.solve(**self.kwargs)
		
		return np.array(H.value).reshape(len(self),len(self))
				
	
	def _build_lipschitz_matrix_param(self, X, fX, grads):
		r""" Use an explicit parameterization
		"""

		# Build the basis
		Es = []
		I = np.eye(len(self))
		for i in range(len(self)):
			ei = I[:,i]
			Es.append(np.outer(ei,ei))
			for j in range(i+1,len(self)):
				ej = I[:,j]
				Es.append(0.5*np.outer(ei+ej,ei+ej))

		alpha = cp.Variable(len(Es))
		H = cp.sum([alpha_i*Ei for alpha_i, Ei in zip(alpha, Es)])
		constraints = [H >> 0]
		
		# Construct gradient constraints
		for grad in grads:
			constraints.append( H >> np.outer(grad, grad))
		
		# Construct linear inequality constraints for samples
		A = np.zeros( (len(X)*(len(X)-1)//2, len(Es)) )
		b = np.zeros(A.shape[0])
		row = 0
		for i in range(len(X)):
			for j in range(i+1,len(X)):
				p = X[i] - X[j]
				A[row, :] = [p.dot(E.dot(p)) for E in Es]
				b[row] = (fX[i] - fX[j])**2
				row += 1

		if A.shape[0] > 0:	
			constraints.append( b <= alpha.__rmatmul__(A) )
		
		problem = cp.Problem(cp.Minimize(cp.sum(alpha)), constraints)
		problem.solve(**self.kwargs)

		alpha = np.array(alpha.value)
		H = np.sum([ alpha_i * Ei for alpha_i, Ei in zip(alpha, Es)], axis = 0)
		return H

	def _build_lipschitz_matrix_cvxopt(self, X, fX, grads):
		r""" Directly accessing cvxopt rather than going through CVXPY results in noticable speed improvements
		"""	
		# Build the basis
		Es = []
		I = np.eye(len(self))
		for i in range(len(self)):
			ei = I[:,i]
			Es.append(np.outer(ei,ei))
			for j in range(i+1,len(self)):
				ej = I[:,j]
				Es.append(0.5*np.outer(ei+ej,ei+ej))


		# Constraint matrices for CVXOPT
		Gs = []
		hs = []

		# Construct linear inequality constraints for samples
		for i in range(len(X)):
			for j in range(i+1,len(X)):
				p = X[i] - X[j]
				G = np.vstack([-p.dot(E.dot(p)) for E in Es]).T
				Gs.append(cvxopt.matrix(G))
				hs.append(cvxopt.matrix( [[ -(fX[i] - fX[j])**2]]))

		# Add constraint to enforce H is positive-semidefinite
		G = cvxopt.matrix(np.vstack([-E.flatten('F') for E in Es]).T)
		Gs.append(G)
		hs.append(cvxopt.matrix(np.zeros((len(self),len(self)))))
	
		# Build constraints 	
		for grad in grads:
			Gs.append(G)
			gg = -np.outer(grad, grad)
			hs.append(cvxopt.matrix(gg))

		# Setup objective	
		c = cvxopt.matrix([ np.trace(E) for E in Es])

		if 'verbose' in self.kwargs:
			cvxopt.solvers.options['show_progress'] = self.kwargs['verbose']

		sol = cvxopt.solvers.sdp(c, Gs = Gs, hs = hs)
		alpha = sol['x']
		H = np.sum([ alpha_i * Ei for alpha_i, Ei in zip(alpha, Es)], axis = 0)
		
		return H


if __name__ == '__main__':
	X = np.random.randn(10,4)
	a = np.random.randn(4,)
	a = np.ones(4,)
	fX = np.dot(X, a).flatten()
	grads = np.tile(a, (X.shape[0], 1))
	lip = LipschitzMatrix()
	#lip._dimension = 4
	#lip._build_lipschitz_matrix_param(X, fX, grads)
	lip.fit(grads = grads)
	#print(lip.H)
	#print(lip.L)
	#lip.shadow_plot(X = X, fX = fX)
	#act = ActiveSubspace(grads)
	#act.shadow_plot(X, fX)
	#plt.show()
