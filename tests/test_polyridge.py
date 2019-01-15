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
	pra.set_scale(X, U)
	res = lambda U: pra._varpro_residual(X, fX, U)
	jac = lambda U: pra._varpro_jacobian(X, fX, U)

	err = check_jacobian(U_flat, res, jac)	
	assert err < 1e-6
	

	# Check with scaling on for Hermite basis
	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension = n, scale = True, basis = 'hermite')
	pra.set_scale(X, U)
	res = lambda U: pra._varpro_residual(X, fX, U)
	jac = lambda U: pra._varpro_jacobian(X, fX, U)

	err = check_jacobian(U_flat, res, jac)	
	assert err < 1e-6


def test_minimax_gradient():
	np.random.seed(1)
	M = 50
	m = 5
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

	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension = n, scale = False)
	#pra.set_scale(X, U)
	#pra._fit_fixed_U_inf_norm(X, fX, U)
	#c = pra.coef	
	c = np.random.randn(len(pra.basis))	

	U_c = np.hstack([U.flatten(), c])

	res = lambda U_c: pra._residual(X, fX, U_c)
	jac = lambda U_c: pra._jacobian(X, fX, U_c)
	
	print res(U_c)
	print jac(U_c)

	err = check_jacobian(U_c, res, jac)	
	assert err < 1e-6

def test_exact():
	np.random.seed(1)
	M = 100
	m = 10
	n = 2
	p = 5
	
	# Samples
	X = np.random.randn(M,m)
	
	# Synthetic function
	a = np.random.randn(m)
	b = np.random.randn(m)
	fX = np.dot(a,X.T)**2 + np.dot(b, X.T)**3

	# Random point
	U, _ = np.linalg.qr(np.random.randn(m,n))
	# Actual ridge subspace
	#U, _ = np.linalg.qr(np.vstack([a,b]).T)	

	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension = n, scale = True)
	pra.fit(X, fX, U0 = U, verbose = 1)
	# Because the data is an exact ridge function, we should (I think) converge to the global solution
	for fX1, fX2 in zip(pra(X), fX):
		print "%10.5e  %10.5e" % (fX1,fX2)
	assert np.all(np.isclose(pra(X), fX))
	
