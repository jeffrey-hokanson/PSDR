from psdr.polyridge_vec import _varpro_residual, _varpro_jacobian, _grassmann_trajectory
from psdr.polyridge_vec import *
from polyrat import LegendrePolynomialBasis, ArnoldiPolynomialBasis
import numpy as np

import pytest

from checkder import check_jacobian 


def test_residual():
	U = np.random.randn(5,2)
	X = np.random.uniform(-1,1, size = (100, 5))
	nout = 3
	fX = np.random.randn(100,nout)
	Basis = LegendrePolynomialBasis
	Basis = ArnoldiPolynomialBasis
	degree = 5

	r = _varpro_residual(U, X, fX, Basis, degree)
	r2 = np.hstack([_varpro_residual(U, X, fXi, Basis, degree) for fXi in fX.T])
	err = np.linalg.norm(r - r2) 
	print(err)
	assert err < 1e-10


@pytest.mark.parametrize("m", [5])
@pytest.mark.parametrize("n", [1,2,3])
@pytest.mark.parametrize("nout", [1,2,3])
@pytest.mark.parametrize("Basis", [LegendrePolynomialBasis, ArnoldiPolynomialBasis])
@pytest.mark.parametrize("degree", [2,3,4])
def test_jacobian(m, n, nout, Basis, degree):
	np.random.seed(0)
	M = 1000
	U = np.random.randn(m,n)
	X = np.random.uniform(-1,1, size = (M, m))
	fX = np.random.randn(M,nout)

	r = _varpro_residual(U, X, fX, Basis, degree)
	J = _varpro_jacobian(U, X, fX, Basis, degree)

	res = lambda u: _varpro_residual(u.reshape(m,n), X, fX, Basis, degree)[:M]
	jac = lambda u: _varpro_jacobian(u.reshape(m,n), X, fX, Basis, degree)[:M,:]
 
	err = check_jacobian(U.flatten(), res, jac, hvec = [1e-6]) 
	assert err < 1e-5


@pytest.mark.parametrize("m", [5,10, 20])
@pytest.mark.parametrize("n", [1,2,3,4])
def test_grassmann_trajectory(m,n):
	U = np.random.randn(m,n)
	U, _ = np.linalg.qr(U)
	Delta = np.random.randn(m,n)
	Delta = Delta - U @ (U.T @ Delta)
	
	for t in np.linspace(0,1, 10):
		Ut = _grassmann_trajectory(U, Delta, t)
		err_orth = np.linalg.norm(Ut.T @ Ut - np.eye(n), 'fro') 
		print(err_orth)
		assert err_orth < 1e-10


def test_polynomial_ridge_approximation():
	M = 1000
	m = 10
	nout = 2
	X = np.random.randn(M, m)
	fX = np.random.randn(M, nout)
	dim = 3
	degree = 5

	polynomial_ridge_approximation(X, fX, dim, degree, verbose = True)
	 
	

if __name__ == '__main__':
	#test_residual()
	#test_jacobian(5, 2, 1, LegendrePolynomialBasis, 5)
	#test_grassmann_trajectory(5,2)
	test_polynomial_ridge_approximation()
	
