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

	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension = n)
	res = lambda U: pra._varpro_residual(X, fX, U)
	jac = lambda U: pra._varpro_jacobian(X, fX, U)

	err = check_jacobian(U_flat, res, jac)	

	assert err < 1e-6
