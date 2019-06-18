from __future__ import division, print_function

import numpy as np

from subspace import SubspaceBasedDimensionReduction

class OuterProductGradient(SubspaceBasedDimensionReduction):
	r"""


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
	def __init__(self, kernel = 'gaussian'):
	
		# TODO: Bandwidth determination strategies
		assert kernel in ['gaussian',], "Invalid kernel provided"
		self.kernel = kernel

	def fit(self, X, fX):
		r""" 
		"""

		X = np.atleast_2d(X)
		assert len(fX.shape) == 1, "Must provide a scalar quantity of interest fX"
		assert len(fX) == len(X), "Number of points X must match number of function samples fX"
		
		N = len(X)
		
		# Step 1: Standardize data
		mean = np.mean(X, axis = 0)
		std = np.std(X, axis = 0)
		Z = (X - mean)/std

		# Step 2: Setup the kernel

		# Bandwidth from Xia 2007, [Li18, eq. 11.5] 
		bw = 2.34*len(X)**(-1./(max(X.shape[1], 3) +6))
		kernel = lambda dist: np.exp(-bw*dist**2/2.)
	
		# Step 3: Estimate gradients
		z_grads = np.zeros(Z.shape)
		for i, zi in enumerate(Z):
			A = np.zeros((X.shape[1]+1, X.shape[1]+1))
			b = np.zeros((X.shape[1]+1))
			for j, zj in enumerate(Z):
				h = np.hstack([1, zj - zi])
				kern = kernel(np.linalg.norm(zi - zj))
				A += np.outer(h, h)*kern
				b += h*fX[j]*kern
			xx = np.linalg.solve(A, b)
			z_grads[i] = xx[1:]

		U, s, VT = np.linalg.svd(z_grads)
		print(VT)	
