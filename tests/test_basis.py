from __future__ import print_function

import numpy as np
import scipy.linalg
from psdr import (MonomialTensorBasis, 
	LegendreTensorBasis, 
	ChebyshevTensorBasis, 
	LaguerreTensorBasis, 
	HermiteTensorBasis)

from checkder import check_jacobian, check_hessian 


def test_equivalence(m = 3, p = 5):
	""" Check that these bases all express the same thing 
	by checking the range of their Vandermonde matrices coincide
	"""
	M = 100
	X = np.random.randn(M, m)
	bases = [MonomialTensorBasis(m, p),
		LegendreTensorBasis(m, p),
		ChebyshevTensorBasis(m, p),
		LaguerreTensorBasis(m, p),
		HermiteTensorBasis(m, p),
		]

	Vs = [basis.V(X) for  basis in bases]
	Us = [scipy.linalg.orth(V) for V in Vs]
	
	for U1 in Us:
		for U2 in Us:
			print(scipy.linalg.svdvals(U1.T.dot(U2)))
			assert np.all(np.isclose(scipy.linalg.svdvals(U1.T.dot(U2)), 1.))

def test_VC(m = 3, p = 5):
	M = 100
	X = np.random.randn(M, m)
	bases = [MonomialTensorBasis(m, p),
		LegendreTensorBasis(m, p),
		ChebyshevTensorBasis(m, p),
		LaguerreTensorBasis(m, p),
		HermiteTensorBasis(m, p),
		]

	for basis in bases:
		V = basis.V(X)
		# Check vector multiplication
		c = np.random.randn(V.shape[1])
		assert np.all(np.isclose(V.dot(c), basis.VC(X, c)))	

		# Check matrix multplication
		c = np.random.randn(V.shape[1], 2)
		assert np.all(np.isclose(V.dot(c), basis.VC(X, c)))	
		
		# Check matrix multplication with a zero
		c[1,:] = 0.
		assert np.all(np.isclose(V.dot(c), basis.VC(X, c)))	


def test_der(m = 3, p = 5, M = 10):
	X = np.random.randn(M, m)
		

	bases = [MonomialTensorBasis(m, p),
		LegendreTensorBasis(m, p),
		ChebyshevTensorBasis(m, p),
		LaguerreTensorBasis(m, p),
		HermiteTensorBasis(m, p),
		]

	for basis in bases:
		obj = lambda x: basis.V(x.reshape(1,-1)).reshape(-1)
		grad = lambda x: basis.DV(x.reshape(1,-1)).reshape(-1, m)
		assert check_jacobian(X[0], obj, grad) < 1e-7
		
		basis.set_scale(X)
		obj = lambda x: basis.V(x.reshape(1,-1)).reshape(-1)
		grad = lambda x: basis.DV(x.reshape(1,-1)).reshape(-1, m)
		assert check_jacobian(X[0], obj, grad) < 1e-7

def test_hessian(m = 2, p = 5):
	X = np.random.randn(10, m)
		

	bases = [MonomialTensorBasis(m, p),
		LegendreTensorBasis(m, p),
		ChebyshevTensorBasis(m, p),
		LaguerreTensorBasis(m, p),
		HermiteTensorBasis(m, p),
		]

	for basis in bases:
		for i in range(len(basis)):
			print("i", i)
			obj = lambda x: basis.V(x.reshape(1,-1))[0,i]
			hess = lambda x: basis.DDV(x.reshape(1,-1))[0,i]
			assert check_hessian(X[0], obj, hess) < 5e-5
		
		
		basis.set_scale(X)
		for i in range(len(basis)):
			print("i", i)
			obj = lambda x: basis.V(x.reshape(1,-1))[0,i]
			hess = lambda x: basis.DDV(x.reshape(1,-1))[0,i]
			assert check_hessian(X[0], obj, hess) < 5e-5

