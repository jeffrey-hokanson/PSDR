import numpy as np
from psdr import PolynomialRidgeApproximation
from checkder import check_jacobian


def test_varpro_jacobian():
	np.random.seed(1)
	M = 100
	m = 10
	n = 2
	p = 5
	
	# Samples
	X = np.random.uniform(-1,1, size = (M,m))
	
	# Synthetic function
	a = np.random.randn(m)
	b = np.random.randn(m)
	fX = np.dot(a,X.T)**2 + np.dot(b, X.T)**3

	# Random point
	U, _ = np.linalg.qr(np.random.randn(m,n))

	U_flat = U.flatten()

	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension = n, scale = False)
	res = lambda U: pra._varpro_residual(X, fX, U)
	jac = lambda U: pra._varpro_jacobian(X, fX, U)

	err = check_jacobian(U_flat, res, jac)	

	assert err < 1e-6
	

	# Check with scaling on
	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension = n, scale = True)
	pra._set_scale(X, U)
	res = lambda U: pra._varpro_residual(X, fX, U)
	jac = lambda U: pra._varpro_jacobian(X, fX, U)

	err = check_jacobian(U_flat, res, jac)	
	assert err < 1e-6
	

	# Check with scaling on for Hermite basis
	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension = n, scale = True, basis = 'hermite')
	pra._set_scale(X, U)
	res = lambda U: pra._varpro_residual(X, fX, U)
	jac = lambda U: pra._varpro_jacobian(X, fX, U)

	err = check_jacobian(U_flat, res, jac)	
	assert err < 1e-6


def test_scaling():
	M = 100
	m = 10
	n = 1
	p = 5

	X = np.random.randn(M,m)
	U = np.random.randn(m, n)
	U, _ = np.linalg.qr(U)

	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension = n, scale = True)
	pra._set_scale(X, U)
	Y = pra._UX(X, U)
	print np.min(Y, axis = 0)
	print np.max(Y, axis = 0)
	assert np.all(np.isclose(np.min(Y, axis = 0), -1)) 
	assert np.all(np.isclose(np.max(Y, axis = 0), 1)) 
