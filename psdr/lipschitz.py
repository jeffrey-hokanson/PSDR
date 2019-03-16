# Subspace based dimension reduction techniques
from __future__ import division, print_function
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import cvxpy as cp
import cvxopt

from subspace import SubspaceBasedDimensionReduction

__all__ = ['LipschitzMatrix']

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
	def __init__(self, method = 'cvxopt', **kwargs):
		self._U = None
		self._L = None
		self.kwargs = kwargs

		if 'solver' not in self.kwargs:
			self.kwargs['solver'] = cp.CVXOPT
		
		if method == 'cvxopt':
			self._build_lipschitz_matrix = self._build_lipschitz_matrix_cvxopt
		elif method == 'param':	
			self._build_lipschitz_matrix = self._build_lipschitz_matrix_param
		elif method == 'cvxpy':	
			self._build_lipschitz_matrix = self._build_lipschitz_matrix_cvxpy
		else:
			raise NotImplementedError

		if 'abstol' not in kwargs:
			self.kwargs['abstol'] = 1e-7
		if 'reltol' not in kwargs:
			self.kwargs['reltol'] = 1e-6
		if 'feastol' not in kwargs:
			self.kwargs['feastol'] = 1e-7
		if 'refinement' not in kwargs:
			self.kwargs['refinement'] = 1

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
		
		H = self._build_lipschitz_matrix(X, fX/scale, grads/scale)
		#self._H = H = scale**2 * H

		# Compute the important directions
		#self._U, _, _ = np.linalg.svd(self._H)
		ew, U = scipy.linalg.eigh(H)
		# because eigenvalues are in ascending order, the subspace basis needs to be flipped
		self._U = U[:,::-1]

		# Force to be SPD
		self._H = scale**2 * U.dot(np.diag(np.maximum(ew,0)).dot(U.T))

		# Compute the Lipschitz matrix (lower triangular)
		#self._L = scipy.linalg.cholesky(self.H[::-1][:,::-1], lower = False)[::-1][:,::-1]
		self._L = scale * U.dot(np.diag(np.sqrt(np.maximum(ew, 0))).dot(U.T))

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

	def _build_lipschitz_matrix_cvxpy(self, X, fX, grads):
		# Constrain H to symmetric positive semidefinite (PSD)
		H = cp.Variable( (len(self), len(self)), PSD = True)
		
		constraints = []		

		# Sample constraint	
		for i in range(len(X)):
			for j in range(i+1, len(X)):
				lhs = (fX[i] - fX[j])**2
				y = X[i] - X[j]
				# y.T M y
				#rhs = H.__matmul__(y).__rmatmul__(y.T)
				rhs = cp.quad_form(y, H)
				constraints.append(lhs <= rhs)
			
		# gradient constraints
		for g in grads:
			constraints.append( np.outer(g,g) << H)

		problem = cp.Problem(cp.Minimize(cp.trace(H)), constraints)
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

		Eten = np.array(Es)

		# Constraint matrices for CVXOPT
		# The format is 
		# sum_i x_i * G[i].reshape(square matrix) <= h.reshape(square matrix)
		Gs = []
		hs = []

		# Construct linear inequality constraints for samples
		for i in range(len(X)):
			for j in range(i+1,len(X)):
				p = X[i] - X[j]
				# Normalizing here seems to reduce the normalization once inside CVXOPT
				p_norm = np.linalg.norm(p)	
				# Vectorize to improve performance
				#G = [-p.dot(E.dot(p)) for E in Es]
				G = -np.tensordot(np.tensordot(Eten, p/p_norm, axes = (2,0)), p/p_norm, axes = (1,0))

				Gs.append(cvxopt.matrix(G).T)
				hs.append(cvxopt.matrix( [[ -(fX[i] - fX[j])**2/p_norm**2]]))

		# Add constraint to enforce H is positive-semidefinite
		# Flatten in Fortran---column major order
		G = cvxopt.matrix(np.vstack([E.flatten('F') for E in Es]).T)
		Gs.append(-G)
		hs.append(cvxopt.matrix(np.zeros((len(self),len(self)))))
	
		# Build constraints 	
		for grad in grads:
			Gs.append(-G)
			gg = -np.outer(grad, grad)
			hs.append(cvxopt.matrix(gg))

		# Setup objective	
		c = cvxopt.matrix(np.array([ np.trace(E) for E in Es]))
		
		if 'verbose' in self.kwargs:
			cvxopt.solvers.options['show_progress'] = self.kwargs['verbose']
		else:
			cvxopt.solvers.options['show_progress'] = False

		for name in ['abstol', 'reltol', 'feastol', 'refinement']:
			if name in self.kwargs:
				cvxopt.solvers.options[name] = self.kwargs[name]

		sol = cvxopt.solvers.sdp(c, Gs = Gs, hs = hs)
		alpha = sol['x']
		H = np.sum([ alpha_i * Ei for alpha_i, Ei in zip(alpha, Es)], axis = 0)
		return H


if __name__ == '__main__':
	import time
	X = np.random.randn(100,6)
	a = np.random.randn(6,)
	a = np.ones(6,)
	fX = np.dot(X, a).flatten()
	grads = np.tile(a, (X.shape[0], 1))
	lip = LipschitzMatrix(method = 'cvxopt')
	#lip._dimension = 4
	#lip._build_lipschitz_matrix_param(X, fX, grads)
	start_time = time.clock()
	#lip.fit(X, fX)
	lip.fit(grads = grads)
	stop_time = time.clock()
	print("Time: ", stop_time - start_time)
	#print(lip.H)
	#print(lip.L)
	#lip.shadow_plot(X = X, fX = fX)
	#act = ActiveSubspace(grads)
	#act.shadow_plot(X, fX)
	#plt.show()

