# Subspace based dimension reduction techniques
from __future__ import division, print_function

import os
if 'DISPLAY' not in os.environ:
	import matplotlib
	matplotlib.use("Agg")
import matplotlib.pyplot as plt


import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist
import cvxpy as cp
import cvxopt
from itertools import combinations


from tqdm import tqdm
import time

#from .domains import Domain	
from .misc import merge
from .subspace import SubspaceBasedDimensionReduction
from .coord import CoordinateBasedDimensionReduction
from .sample import initial_sample
from .geometry import voronoi_vertex_sample
from .geometry import unique_points
from .minimax import minimax
from .function import BaseFunction
from .pgf import PGF

__all__ = ['LipschitzMatrix', 'LipschitzConstant', 'DiagonalLipschitzMatrix']


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

	If the parameter :code:`epsilon` is passed, the :math:`\epsilon`-Lipschitz matrix
	is computed instead by solving the semidefinite program
	
	.. math::

		\min_{\mathbf{H} \in \mathbb{S}^{m}_+} & \ \text{Trace } \mathbf{H} \\
		\text{such that} & \ (|f(\mathbf{x}_i) - f(\mathbf{x}_j)| -\epsilon)^2 
			\le (\mathbf{x}_i - \mathbf{x}_j)^\top \mathbf{H} (\mathbf{x}_i - \mathbf{x}_j).

	Note that when computing the :math:`\epsilon`-Lipschitz matrix,
	gradient information cannot be used because gradients no longer constrain 
	the Lipschitz matrix.

	Parameters
	----------
	epsilon: float or None
		Tolerance to be used if find the :math:`\epsilon`-Lipschitz matrix
	**kwargs: dict (optional)
		Additional parameters to pass to cvxpy
	"""
	def __init__(self, epsilon = None, method = 'cvxopt', L = None, **kwargs):
		if L is not None:
			self._L = np.atleast_2d(L)
		self._U = None
		self.kwargs = kwargs

		assert method in ['cvxopt', 'param', 'cvxpy']
		if method == 'cvxopt':
			self._build_lipschitz_matrix = self._build_lipschitz_matrix_cvxopt
		elif method == 'param':	
			self._build_lipschitz_matrix = self._build_lipschitz_matrix_param
		elif method == 'cvxpy':	
			self._build_lipschitz_matrix = self._build_lipschitz_matrix_cvxpy


		if epsilon is not None:
			epsilon = float(epsilon)
			assert epsilon >= 0, "Epsilon must be positive"
		self.epsilon = epsilon
		
		# According to a google groups discussion
		# https://groups.google.com/forum/#!topic/cvxopt/HSc80A5mWzY
		# Too small tolerances can yield divide by zero errors
		# 1e-10 was breaking, using 1e-9 instead
		self.kwargs = merge({'solver': 'CVXOPT', 'abstol':1e-9, 'reltol':1e-9, 'feastol':1e-9, 'refinement': 1, 'kktsolver': 'robust'} , kwargs)

	def __str__(self):
		if self.epsilon is not None and self.epsilon > 0:
			return "<Lipschitz Matrix epsilon %g>" % (self.epsilon)
		return "<Lipschitz Matrix>"

	def fit(self, X = None, fX = None, grads = None):
		r""" Estimate the Lipschitz matrix from data


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
	
		# Scale the output units for numerical stability	
		try:	
			scale1 = np.max(fX) - np.min(fX)
		except ValueError:
			scale1 = 1.
		try:
			scale2 = np.max([np.linalg.norm(grad) for grad in grads])
		except ValueError:
			scale2 = 1.
		scale = max(scale1, scale2)
	
		if self.epsilon is not None:
			epsilon = self.epsilon/scale
		else:
			epsilon = 0.

		self._fit(X, fX, grads, epsilon, scale)


	def _fit(self, X, fX, grads, epsilon, scale):
		H = self._build_lipschitz_matrix(X, fX/scale, grads/scale, epsilon)

		# Compute the important directions
		#self._U, _, _ = np.linalg.svd(self._H)
		ew, U = scipy.linalg.eigh(H)

		# Because eigenvalues are in ascending order, the subspace basis needs to be flipped
		U = U[:,::-1]
		ew = ew[::-1]
		# Fix the signs for the subspace directions 
		U = self._fix_subspace_signs(U, X, fX/scale, grads/scale)
		self._U = U	
		# Force to be SPD
		self._H = scale**2 * U.dot(np.diag(np.maximum(ew,0)).dot(U.T))

		# Compute the Lipschitz matrix 
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

	def _build_lipschitz_matrix_cvxpy(self, X, fX, grads, epsilon):
		# Constrain H to symmetric positive semidefinite (PSD)
		H = cp.Variable( (len(self), len(self)), PSD = True)
		
		constraints = []		

		# Sample constraint	
		for i in range(len(X)):
			for j in range(i+1, len(X)):
				lhs = (np.abs(fX[i] - fX[j]) - epsilon)**2
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
		print("Inside Lipschitz")
		print(np.array(H.value))	
		return np.array(H.value).reshape(len(self),len(self))
				
	
	def _build_lipschitz_matrix_param(self, X, fX, grads, epsilon):
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
				b[row] = (np.abs(fX[i] - fX[j]) - epsilon)**2
				row += 1

		if A.shape[0] > 0:	
			constraints.append( b <= alpha.__rmatmul__(A) )
		
		problem = cp.Problem(cp.Minimize(cp.sum(alpha)), constraints)
		problem.solve(**self.kwargs)

		alpha = np.array(alpha.value)
		H = np.sum([ alpha_i * Ei for alpha_i, Ei in zip(alpha, Es)], axis = 0)
		return H

	def _build_lipschitz_matrix_cvxopt(self, X, fX, grads, epsilon, structure = 'full'):
		r""" Directly accessing cvxopt rather than going through CVXPY results in noticable speed improvements
		"""	
		# Build the basis
		Es = []
		I = np.eye(len(self))
	
		if structure == 'full':
			for i in range(len(self)):
				ei = I[:,i]
				Es.append(np.outer(ei,ei))
				for j in range(i+1,len(self)):
					ej = I[:,j]
					Es.append(0.5*np.outer(ei+ej,ei+ej))
		elif structure == 'diag':
			for i in range(len(self)):
				ei = I[:,i]
				Es.append(np.outer(ei,ei))
		else:
			raise NotImplementedError	

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
				hs.append(cvxopt.matrix( [[ -(np.abs(fX[i] - fX[j]) - epsilon)**2/p_norm**2]]))

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

		# For some reason, using the 'kktsolver' option fails
		for name in ['abstol', 'reltol', 'feastol', 'refinement']:
			if name in self.kwargs:
				cvxopt.solvers.options[name] = self.kwargs[name]

		sol = cvxopt.solvers.sdp(c, Gs = Gs, hs = hs)
		alpha = sol['x']
		H = np.sum([ alpha_i * Ei for alpha_i, Ei in zip(alpha, Es)], axis = 0)
		return H

	def uncertainty(self, X, fX, Xtest):
		r""" Compute range of possible values at test points.

		Given pairs of inputs :math:`\widehat{\mathbf{x}}_i`
		and outputs  :math:`y_i := f(\widehat{\mathbf{x}}_i)`
		as well as the Lipschitz matrix :math:`\mathbf{L}`, compute the uncertainty interval
		associated with the provided points :math:`\mathbf{x}_i`; namely,

		.. math::
		
			\mathcal{U}(\mathbf{x}) = 
			\left[ 
				\max_{j=1,\ldots,M} y_j - \|\mathbf{L} (\mathbf{x} - \widehat{\mathbf{x}}_i)\|_2, \
				\min_{j=1,\ldots,M} y_j + \|\mathbf{L} (\mathbf{x} - \widehat{\mathbf{x}}_i)\|_2
			\right].

		Parameters
		----------
		X: array-like (M, m)
			Array of input coordinates
		fX: array-like (M,)
			Array of function values
		Xtest: array-like (N, m)
			Array of places at which to determine uncertainty

		Returns
		-------
		lb: array-like (N,)
			Lower bounds
		ub: array-like (N,)
			Upper bounds
		"""
		X = np.atleast_2d(np.array(X))
		Xtest = np.atleast_2d(np.array(Xtest))

		lb = -np.inf*np.ones(Xtest.shape[0])
		ub = np.inf*np.ones(Xtest.shape[0])

		fX = fX.flatten()

		LX = self.L.dot(X.T).T
		LXtest = self.L.dot(Xtest.T).T
	
		# TODO: Replace with proper chunking to help vectorize this operation	
		for Lx, fx in zip(LX, fX):
			dist = cdist(Lx.reshape(1,-1), LXtest).flatten()
			lb = np.maximum(lb, fx - dist)
			ub = np.minimum(ub, fx + dist)
		
		return lb, ub
	
	def uncertainty_domain(self, X, fX, domain, Nsamp = int(1e2), verbose = False, progress = False, tqdm_kwargs = {}, **kwargs):
		r""" Compute the uncertainty associated with a set inside the domain
		
		This estimates the uncertainty associated with a subset of the domain,
		essentially finding points :math:`\mathbf{x} \in \mathcal{S}`
		corresponding to extrema of the uncertainty interval

		.. math::
		
			\mathcal{U}(\mathcal{S} ) = 
			\left[ 
				\min_{\mathbf{x} \in \mathcal{S}} \max_{j=1,\ldots,M} y_j - \|\mathbf{L} (\mathbf{x} - \widehat{\mathbf{x}}_i)\|_2, \
				\max_{\mathbf{x} \in \mathcal{S}} \min_{j=1,\ldots,M} y_j + \|\mathbf{L} (\mathbf{x} - \widehat{\mathbf{x}}_i)\|_2
			\right].

		Parameters
		----------
		X: array-like (M, m)
			Array of input coordinates
		fX: array-like (M,)
			Array of function values
		domain: Domain
			Domain on which to find exterma of limits
		Nsamp: int, optional
			Number of Voronoi vertices to sample to find exterma
		verbose: bool (default: False)
			If True, print the iteration history for the optimization program for each initalization
		progress: bool (default: False)
			If True, show a progress bar for trying different initializations
		tqdm_kwargs: dict, optional
			Additional arguments to pass to tqdm for progress plotting

		Returns
		-------
		lb: float
			Lower bound
		ub: float
			Upper bound
		"""
		lower_bound = LowerBound(self.L, X, fX)
		upper_bound = UpperBound(self.L, X, fX)
		
		fX = fX.flatten()
	
		# Force these to be far apart
		X0 = initial_sample(domain, self.L, Nsamp = Nsamp) 
		X0 = voronoi_vertex_sample(domain, X, X0, L = self.L, randomize = False)

		# Remove duplicates
		I = unique_points(X0)
		X0 = X0[I]


		# Iterate through all candidates
		if progress:
			iterator = tqdm(X0, desc = 'bounds_domain', total = len(X0), dynamic_ncols = True, **tqdm_kwargs)
		else:
			iterator = X0

		lbs = []
		ubs = []
		for x0 in iterator:
			# Lower bound
			x = minimax(lower_bound, x0, domain = domain, verbose = verbose, trust_region = False)
			lbs.append(np.max(lower_bound(x)))
		
			# Upper bound
			x = minimax(upper_bound, x0, domain = domain, verbose = verbose, trust_region = False)
			ubs.append(np.min(-upper_bound(x)))
			#print(lbs[-1], ubs[-1])
		return float(np.min(lbs)), float(np.max(ubs))

#		# Attempts at making a parallel interface
#		else:
#			lbs = []
#			ubs = []
#			for x0 in X0:
#				x = dask.delayed(minimax)(lower_bound, x0, domain = domain, trust_region = False)
#				lb = dask.delayed(lower_bound)(x)
#				lbs.append(lb)
#
#				x = dask.delayed(minimax)(upper_bound, x0, domain = domain, trust_region = False)
#				ub = dask.delayed(upper_bound)(x)
#				ubs.append(ub)
#				
#			lb = dask.delayed(np.max)(lbs)
#			ub = dask.delayed(np.max)(ubs)
#
#			with ProgressBar():
#				lb = lb.compute()
#				ub = -1.*ub.compute()
#			return lb, ub
#			if client is None:
#				client = Client(processes = False)
#			lb_res = []
#			ub_res = []
#			for x0 in X0:
#				res = client.submit(minimax, lower_bound, x0, domain = domain, trust_region = False)
#				lb_res.append(res)
#				res = client.submit(minimax, upper_bound, x0, domain = domain, trust_region = False)
#				ub_res.append(res)
#		
#			if progress:
#				with tqdm(range(len(X0)), desc = 'bounds_domain', total = 2*len(X0),
#					 dynamic_ncols = True, **tqdm_kwargs) as it:
#					while True:
#						it.n = np.sum([res.done() for res in lb_res + ub_res])
#						it.refresh()
#						if it.n == 2*len(X0):
#							break
#						time.sleep(0.1)
#
#			lbs = np.array([np.max(lower_bound(res.result())) for res in lb_res])	
#			ubs = np.array([np.min(-upper_bound(res.result())) for res in ub_res])	
#


	def shadow_uncertainty(self, domain, X, fX, ax = None, ngrid = 50, dim = 1, U = None, pgfname = None,
			plot_kwargs = {}, progress = False, **kwargs):
		r""" Plot the uncertainty intervals given data
		"""
		X = np.atleast_2d(np.array(X))
		fX = np.array(fX)


		assert dim == 1, "Higher dimensions not yet implemented"

		if U is None:
			u = self.U[:,0].flatten()
		else:
			u = U.flatten()

		# Find extent in the active direction
		c1 = domain.corner(u)
		e1 = u.dot(c1)
		c2 = domain.corner(-u)
		e2 = u.dot(c2)

		# Flip so order is increasing
		if e1 > e2:
			e1, e2 = e2, e1
			c1, c2 = c2, c1

		# Points to estimate uncertainty at
		ys = np.linspace(e1, e2, ngrid)
		lbs = np.zeros(ngrid)
		ubs = np.zeros(ngrid)

		# We treat the endpoints separately because there is only one point
		lbs[0], ubs[0] = self.uncertainty(X, fX, c1)
		lbs[-1], ubs[-1] = self.uncertainty(X, fX, c2)

		# Now for the points on the interior
		iterator = range(1, ngrid - 1)

		tqdm_kwargs = {}
		if progress:
			iterator = tqdm(iterator, "shadow uncertainty", position = 0, dynamic_ncols = True)
			if progress > 1:
				tqdm_kwargs = {'position':1}

		for i in iterator:
			dom_eq = domain.add_constraints(A_eq = u.reshape(1,-1), b_eq = ys[i].reshape(1))
			lbs[i], ubs[i] = self.uncertainty_domain(X, fX, dom_eq, progress = progress - 1, tqdm_kwargs = tqdm_kwargs, **kwargs)

		if ax is not None:
			ax.fill_between(ys, lbs, ubs, **plot_kwargs)	
	
		if pgfname is not None:
			pgf = PGF()
			pgf.add('y', ys)
			pgf.add('lb', lbs)
			pgf.add('ub', ubs)
			pgf.write(pgfname)

class DiagonalLipschitzMatrix(CoordinateBasedDimensionReduction, LipschitzMatrix):
	r""" Constructs a diagonal Lipschitz matrix

	Much like the standard Lipschitz matrix class :meth:`psdr.LipschitzMatrix`,
	this class computes a Lipschitz matrix except with the constraint the matrix is diagonal.
	"""

	def __init__(self, L = None, **kwargs):
		if L is not None:
			L = np.atleast_2d(L)
			self._L = np.diag(np.diag(L))

		LipschitzMatrix.__init__(self,**kwargs)
		

	def _fit(self, X, fX, grads, epsilon, scale):
		H = self._build_lipschitz_matrix(X, fX/scale, grads/scale, epsilon, structure = 'diag')
		# Ensure entries are numerically positive
		H = np.diag(np.maximum(np.diag(H),0))
		# Unlike the matrix case, we do not rotate coordinates
		self._H = scale**2 * H
		self._L = scale * np.diag(np.sqrt(np.diag(H)))
		
		self._score = np.diag(self._L).copy()		
		#self._U = np.eye(len(self._score))[:,np.argsort(-self._score)] 

class LipschitzConstant(LipschitzMatrix):
	r""" Computes the scalar Lipschitz constant
	"""
	def __init__(self, L = None, **kwargs):
		if L is not None:
			self._L = float(L)	

		self._dimension = 1	
		LipschitzMatrix.__init__(self, **kwargs)

	def _fit(self, X, fX, grads, epsilon, scale):
		L = 0
		if len(X) > 0:
			L_samp = np.max([ (abs(fX[i] - fX[j]) - epsilon )/ np.linalg.norm(X[i] - X[j]) 
				for i,j in combinations(range(len(X)), 2)])
			L = max(L, L_samp)

		if len(grads) > 0:
			L_grad = np.max([np.linalg.norm(grad) for grad in grads])
			L = max(L, L_grad)

		self._L = L

	@property
	def H(self):
		return self._L**2 * np.eye(len(self))

	@property
	def L(self):
		return self._L * np.eye(len(self))

	@property
	def U(self):
		return np.eye(len(self))

# Helper functions for determining bounds

class LowerBound(BaseFunction):
	def __init__(self, L, X, fX):
		self.L = L
		self.Y = np.dot(self.L, X.T).T
		self.fX = fX.reshape(len(X))

	def eval(self, x):
		y = np.dot(self.L, x)
		norms = cdist(y.reshape(1,-1), self.Y, 'euclidean').flatten()
		return self.fX - norms

	def grad(self, x):
		y = np.dot(self.L, x)
		norms = cdist(y.reshape(1,-1), self.Y, 'euclidean').flatten()
		G = np.zeros([self.Y.shape[0], x.shape[0]] )
		I = (norms > 0)
		G[I,:] = -(  (y - self.Y[I]).T /norms[I]).T
		G = self.L.T.dot(G.T).T
		return G

class UpperBound(BaseFunction):
	def __init__(self, L, X, fX):
		self.L = L
		self.Y = np.dot(self.L, X.T).T
		self.fX = fX.reshape(len(X))

	def eval(self, x):
		y = np.dot(self.L, x)
		norms = cdist(y.reshape(1,-1), self.Y, 'euclidean').flatten()
		return -(self.fX + norms)

	def grad(self, x):
		y = np.dot(self.L, x)
		norms = cdist(y.reshape(1,-1), self.Y, 'euclidean').flatten()
		G = np.zeros([self.Y.shape[0], x.shape[0]] )
		I = (norms > 0)
		G[I,:] = -((y - self.Y[I]).T/norms[I]).T
		G = self.L.T.dot(G.T).T
		return G


