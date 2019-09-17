from __future__ import division, print_function

import numpy as np
from scipy.spatial.distance import cdist
from .subspace import SubspaceBasedDimensionReduction, ActiveSubspace

__all__ = ['OuterProductGradient']

def opg_grads(Z, fZ, kernel = None):
	if kernel is None:
		# Bandwidth from Xia 2007, [Li18, eq. 11.5] 
		bw = 2.34*len(Z)**(-1./(max(Z.shape[1], 3) +6))
		kernel = lambda dist: np.exp(-bw*dist**2/2.)
		 
	
	# Append ones to accelerate 
	M = Z.shape[0]
	Y = np.hstack([np.ones((M, 1)), Z])
	
	# estimated gradients in the transformed coordinates
	z_grads = np.zeros(Z.shape)	

	for i, zi in enumerate(Z):
		# Compute 2-norm distance between points
		d = cdist(Z, zi.reshape(1,-1), 'euclidean').flatten()
		
		weights = kernel(d).reshape(-1,1)
		# This is sum_j weight_j * y_j y_j^T 
		A = Y.T.dot(weights*Y)
		# This is sum_j weight_j * y_j * fX_j
		b = np.sum((weights*fZ.reshape(-1,1))*Y, axis = 0)
		# Estimate the coefficients of the line
		g = np.linalg.solve(A, b)
		# Extract the slope as the gradient
		z_grads[i] = g[1:]

	return z_grads


class OuterProductGradient(ActiveSubspace):
	r""" The Outer Product Gradient approach of Hardle and Stoker


	Given pairs of inputs :math:`\mathbf{x}_j` and outputs :math:`f(\mathbf{x}_j)`,
	this method estimates the active subspace by (1) estimating gradients from these samples
	and then (2) using the eigendecomposition of the outer product of these approximate gradients 
	The gradients are estimated by fitting a linear model at each input point :math:`\mathbf{x}_j`
	weighting the other points based on a kernel :math:`k` that decreases with increasing distance;
	i.e., at :math:`\mathbf{x}_j` we construct the linear model :math:`\mathbf{a}^\top \mathbf{x} + \beta`: 
	by solving a linear system:

	.. math::
		
		\min_{\mathbf{a}, \beta} 
		\left\|
		\begin{bmatrix}
			\sqrt{k(\| \mathbf{x}_1 - \mathbf{x}_j\|)} & & & \\	
			& \sqrt{k(\| \mathbf{x}_2 - \mathbf{x}_j\|)} & & \\
			& & \ddots & \\
			& & & \sqrt{k(\|\mathbf{x}_M - \mathbf{x}_j\|)}
		\end{bmatrix}
		\left[	
		\begin{bmatrix} 
			\mathbf{x}_1^\top - \mathbf{x}_j^\top & 1 \\ 
			\mathbf{x}_2^\top - \mathbf{x}_j^\top & 1 \\ 
			\vdots & \vdots \\
			\mathbf{x}_M^\top - \mathbf{x}_j^\top & 1 \\ 
		\end{bmatrix}
		\begin{bmatrix} \mathbf{a} \\ \beta \end{bmatrix}
		- 
		\begin{bmatrix}
			f(\mathbf{x}_1) \\ 
			f(\mathbf{x}_2) \\
			\vdots \\
			f(\mathbf{x}_M)
		\end{bmatrix}
		\right]
		\right\|_2.
		

	Our implementation largely follows [Li18]_.

	References
	----------
	.. [HS89]  W. Hardle and T. M. Stoker.
		Investigating Smooth Multiple Regression by the Method of Average Derivatives
		J. Am. Stat. Ass. Vol 84, No 408, (1989). pp 986--995, DOI:0.1080/01621459.1989.10478863  	
	.. [Li18] Bing Li.
		Sufficient Dimension Reduction: Methods and Applications with R.
		CRC Press, 2018.
	"""
	def __init__(self, kernel = 'gaussian', standardize = True):
	
		# TODO: Bandwidth determination strategies
		assert kernel in ['gaussian',], "Invalid kernel provided"
		self.kernel = kernel
		self.standardize = standardize

	def __str__(self):
		return "<Outer Product Gradient>"

	def fit(self, X, fX):
		r""" 

		Parameters
		----------
		X: array-like (M,m)
			Location of function samples
		fX: array-like (M,)
			Function samples where fX[j] = f(X[j])
		"""

		X = np.atleast_2d(X)
		fX = np.atleast_1d(fX)

		if len(fX.shape) > 1:
			fX = fX.flatten()
			assert len(fX) == len(X), "Must provide one-dimensional inputs of same dimension as X"
		
		assert len(fX) == len(X), "Number of points X must match number of function samples fX"
		
		N = len(X)
		
		# Step 1: Standardize data
		if self.standardize:
			mean = np.mean(X, axis = 0)
			std = np.std(X, axis = 0)
			Z = (X - mean)/std
		else:
			Z = X

		# Step 2: Setup the kernel

		# Bandwidth from Xia 2007, [Li18, eq. 11.5] 
		bw = 2.34*len(X)**(-1./(max(X.shape[1], 3) +6))
		kernel = lambda dist: np.exp(-bw*dist**2/2.)

		# TODO: Implement both naive approach in Li as well as weighted LS
		# N.B.: weighted LS will be slower (but better conditioned) because it is solve a large LS
		# vs a small pos-definite linear system

		# Step 3: Estimate gradients
		z_grads = opg_grads(Z, fX, kernel)

		# Step 4: identify the active subspace
		ActiveSubspace.fit(self, z_grads)
			
		if self.standardize:
			# Due to the coordiante change introduced by standardization 
			Dinv2 = np.diag(std**(-0.5))
			U = Dinv2.dot(self.U)
			Q, R = np.linalg.qr(U, mode = 'reduced')
			self._U = Q
			self._U = self._fix_subspace_signs_grads(self._U, z_grads)		
		

