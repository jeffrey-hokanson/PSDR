from __future__ import division, print_function

import numpy as np
from scipy.spatial.distance import cdist
from .subspace import SubspaceBasedDimensionReduction, ActiveSubspace
from .local_linear import local_linear_grads


__all__ = ['OuterProductGradient']


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
		

	Our implementation is a modification of the algorithm presented by Li [Li18]_.
	Gradients are estimated by calling :code:`psdr.local_linear_grad` which uses
	a more numerically stable approach for estimating gradients than that above.
	Additionally, by default we use a per-location bandwidth based on perplexity.	

	Parameters
	----------
	standardize: bool
		If True, standardize input X to have zero mean and identity covariance.
	perplexity: None or float
		Entropy target for choosing bandwidth 
	bandwidth: None, 'xia' or positive float
 		If specified, set a global bandwidth.  'xia' uses the heuristic
		suggested in [Li18]_.

	References
	----------
	.. [HS89]  W. Hardle and T. M. Stoker.
		Investigating Smooth Multiple Regression by the Method of Average Derivatives
		J. Am. Stat. Ass. Vol 84, No 408, (1989). pp 986--995, DOI:0.1080/01621459.1989.10478863  	
	.. [Li18] Bing Li.
		Sufficient Dimension Reduction: Methods and Applications with R.
		CRC Press, 2018.
	"""
	def __init__(self, standardize = True, perplexity = None, bandwidth = None):
		self.standardize = standardize
		self.perplexity = perplexity
		self.bandwidth = bandwidth

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
		if self.bandwidth == 'xia':
			# Bandwidth from Xia 2007, [Li18, eq. 11.5] 
			bandwidth = 2.34*len(X)**(-1./(max(X.shape[1], 3) +6))
		else:
			bandwidth = self.bandwidth

		# Step 3: Estimate gradients
		z_grads = local_linear_grads(Z, fX, perplexity = self.perplexity, bandwidth = self.bandwidth)

		# Step 4: identify the active subspace
		ActiveSubspace.fit(self, z_grads)
			
		if self.standardize:
			# Undo the coordiante change introduced by standardization 
			Dinv2 = np.diag(std**(-0.5))
			U = Dinv2.dot(self.U)
			Q, R = np.linalg.qr(U, mode = 'reduced')
			self._U = Q
			self._U = self._fix_subspace_signs_grads(self._U, z_grads)		
		
