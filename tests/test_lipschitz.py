from __future__ import print_function
import numpy as np
import scipy.linalg
from psdr import LipschitzMatrix
from psdr.demos import OTLCircuit

def test_lipschitz_grad(N = 10):

	func = OTLCircuit()
	X = func.sample(N)
	grads = func.grad(X)
	
	lip = LipschitzMatrix(ftol = 1e-10, gtol = 1e-10, verbose = True)
	lip.fit(grads = grads)
	
	H = np.copy(lip.H)

	for g in grads:
		gap = np.min(scipy.linalg.eigvalsh(H - np.outer(g,g)))
		print(gap)
		assert gap >= -1e-6
		
def test_lipschitz_func(M = 20):

	func = OTLCircuit()
	X = func.sample(M)
	fX = func(X)
	
	lip = LipschitzMatrix(ftol = 1e-10, gtol = 1e-10, verbose = True)
	lip.fit(X, fX)
	
	H = np.copy(lip.H)

	for i in range(M):
		for j in range(i+1,M):
			y = X[i] - X[j]
			gap = y.dot(H.dot(y)) - (fX[i] - fX[j])**2
			print(gap)
			assert gap >= -1e-6
