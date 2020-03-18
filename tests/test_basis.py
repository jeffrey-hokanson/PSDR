from __future__ import print_function

import numpy as np
import scipy.linalg
from psdr import (MonomialTensorBasis, 
	LegendreTensorBasis, 
	ChebyshevTensorBasis, 
	LaguerreTensorBasis, 
	HermiteTensorBasis,
	ArnoldiPolynomialBasis)

from checkder import check_jacobian, check_hessian 


def test_equivalence(m = 3, p = 5):
	""" Check that these bases all express the same thing 
	by checking the range of their Vandermonde matrices coincide
	"""
	np.random.seed(0)
	M = 100
	X = np.random.randn(M, m)
	bases = [MonomialTensorBasis(p, dim=m),
		LegendreTensorBasis(p, dim=m),
		ChebyshevTensorBasis(p, dim=m),
		LaguerreTensorBasis(p, dim=m),
		HermiteTensorBasis(p, dim=m),
		]

	Vs = [basis.V(X) for  basis in bases]
	Us = [scipy.linalg.orth(V) for V in Vs]
	
	for U1 in Us:
		for U2 in Us:
			print(scipy.linalg.svdvals(U1.T.dot(U2)))
			assert np.all(np.isclose(scipy.linalg.svdvals(U1.T.dot(U2)), 1.))

def test_VC(m = 3, p = 5):
	np.random.seed(0)
	M = 100
	X = np.random.randn(M, m)
	bases = [MonomialTensorBasis(p, dim=m),
		LegendreTensorBasis(p, dim=m),
		ChebyshevTensorBasis(p, dim=m),
		LaguerreTensorBasis(p, dim=m),
		HermiteTensorBasis(p, dim=m),
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
	np.random.seed(0)
	X = np.random.randn(M, m)
		

	bases = [MonomialTensorBasis(p, dim=m),
		LegendreTensorBasis(p, dim=m),
		ChebyshevTensorBasis(p, dim=m),
		LaguerreTensorBasis(p, dim=m),
		HermiteTensorBasis(p, dim=m),
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
	np.random.seed(0)
	X = np.random.randn(10, m)
		

	bases = [MonomialTensorBasis(p, dim=m),
		LegendreTensorBasis(p, dim=m),
		ChebyshevTensorBasis(p, dim=m),
		LaguerreTensorBasis(p, dim=m),
		HermiteTensorBasis(p, dim=m),
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


def test_arnoldi(dim = 2, N = 1000):
	np.random.seed(0)
	X = np.random.rand(N,dim)

	degree = 10
	basis1 = LegendreTensorBasis(degree, dim = dim)
	basis2 = ArnoldiPolynomialBasis(degree, X = X)


	V1 = basis1.V(X)
	V2 = basis2.V(X)
	# Check that V2 is orthonormal
	assert np.max(np.abs(V2.T @ V2 - np.eye(V2.shape[1]))) < 1e-10


	# check basis represents same object
	for k in range(1,V1.shape[1]):
		phi = scipy.linalg.subspace_angles(V1[:,:k], V2[:,:k])
		print(k, np.max(phi))
		assert np.max(phi) < 1e-8, "Subspace angle too large"
	
	# check basis represents same derivatives
	DV1 = basis1.DV(X)
	DV2 = basis2.DV(X)

	for ell in range(X.shape[1]):
		for k in range(2,DV1.shape[1]):
			#print(DV1[:10,:k,ell])
			#print(DV2[:10,:k,ell])
			#print(k, ell)
			phi = scipy.linalg.subspace_angles(DV1[:,:k,ell], DV2[:,:k,ell])
			if len(phi) >0:
				print(k, np.max(phi))
				assert np.max(phi) < 1e-8, "Subspace angle too large"


	# Check when evaluating at new points
	X2 = np.random.rand(2*N, dim)

	print("Checking with a different set of points")	
	V1 = basis1.V(X2)
	V2 = basis2.V(X2)
	for k in range(1,V1.shape[1]):
		phi = scipy.linalg.subspace_angles(V1[:,:k], V2[:,:k])
		print(k, np.max(phi))
		assert np.max(phi) < 1e-8, "Subspace angle too large"


def test_arnoldi_der(dim = 3, N = 1000):
#	np.random.seed(1)
	X = np.random.rand(N,dim)
	X2 = np.random.rand(N+10, dim)
	degree = 10
	basis1 = LegendreTensorBasis(degree, dim = dim)
	basis2 = ArnoldiPolynomialBasis(degree, X = X)

	# Construct on polynomial
	V1 = basis1.V(X)
	V2 = basis2.V(X)
	DV1 = basis1.DV(X2)
	DV2 = basis2.DV(X2)
	for k in range(V1.shape[1]):
		ek = np.zeros(V1.shape[1])
		ek[k] = 1
		fX = V1 @ ek
	
		c2 = V2.T @ fX
		assert np.linalg.norm(fX - V2 @ c2) < 1e-10, "Did not approximate correctly"
		# Check derivative
		
		fXp1 = np.einsum('ijk,j->ik', DV1, ek) 
		fXp2 = np.einsum('ijk,j->ik', DV2, c2) 
		err = np.max(np.abs(fXp1 - fXp2))
		print('ek', k)
		for j in range(1):
			print(fXp1[j,:], '\t', fXp2[j,:], '\t', fXp1[j,:]/fXp2[j,:], '\t', X[j,:])
		print(err)
		assert err < 1e-8


if __name__ == '__main__':
#	test_arnoldi()
	test_arnoldi_der()
