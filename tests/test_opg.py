import numpy as np
from psdr.opg import opg_grads

def opg_grads_old(Z, fZ, kernel = None):
	if kernel is None:
		# Bandwidth from Xia 2007, [Li18, eq. 11.5] 
		bw = 2.34*len(Z)**(-1./(max(Z.shape[1], 3) +6))
		kernel = lambda dist: np.exp(-bw*dist**2/2.)
		 
	z_grads = np.zeros(Z.shape)		# estimated gradients in the transformed coordinates
	
	for i, zi in enumerate(Z):
		# This essentially fits the linear model by solving the normal equations
		A = np.zeros((Z.shape[1]+1, Z.shape[1]+1))
		b = np.zeros((Z.shape[1]+1))
		for j, zj in enumerate(Z):
			h = np.hstack([1, zj - zi])
			# TODO: Li mentions the kernel can be evaluated cheaper when Gaussian
			kern = kernel(np.linalg.norm(zi - zj))
			A += np.outer(h, h)*kern
			b += h*fZ[j]*kern
		xx = np.linalg.solve(A, b)
		z_grads[i] = xx[1:]

	return z_grads

def test_opg_grads(M = 100, m=5):
	Z = np.random.randn(M, m)
	fZ = np.random.randn(M)
	grads1 = opg_grads(Z, fZ)
	grads2 = opg_grads(Z, fZ)

	assert np.all(np.isclose(grads1, grads2))
