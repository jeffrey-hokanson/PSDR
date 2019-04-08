from __future__ import print_function
import numpy as np
import scipy.linalg
from psdr import PolynomialRidgeApproximation, LegendreTensorBasis, PolynomialRidgeFunction
from .checkder import *


def test_affine():
	X = np.random.randn(100, 5)
	a = np.random.randn(5,)
	y = X.dot(a)
		
	pra = PolynomialRidgeApproximation(degree = 1, subspace_dimension = 1, bound = 'upper')
	pra.fit(X, y)
	assert np.all(np.isclose(y, pra(X)))
	
	ang = scipy.linalg.subspace_angles(pra.U, a.reshape(-1,1))
	assert np.isclose(ang, 0)
	
	pra = PolynomialRidgeApproximation(degree = 1, subspace_dimension = 1)
	pra.fit(X, y)
	assert np.all(np.isclose(y, pra(X)))
	
	ang = scipy.linalg.subspace_angles(pra.U, a.reshape(-1,1))
	assert np.isclose(ang, 0)


def test_polyridge_der():
	m = 5
	n = 1
	p = 3
	
	U = scipy.linalg.orth(np.random.randn(m,n))
	coef = np.random.randn(len(LegendreTensorBasis(n,p)))
	prf = PolynomialRidgeFunction(LegendreTensorBasis(n,p), coef, U)

	x = np.random.randn(m)

	print(prf.eval(x))
	print(prf.grad(x))
	print(prf.hessian(x))
	
	assert check_derivative(x, prf.eval, lambda x: prf.grad(x) ) < 1e-7
	assert check_hessian(x, prf.eval, lambda x: prf.hessian(x) ) < 1e-5
		
	

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
	
	print(res(U_c))
	print(jac(U_c))

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
		print("%10.5e  %10.5e" % (fX1,fX2))
	assert np.all(np.isclose(pra(X), fX))

def exact_data(M = 100, m = 10, n = 1, p = 3):
	U = scipy.linalg.orth(np.random.randn(m,n))
	coef = np.random.randn(len(LegendreTensorBasis(n,p)))
	prf = PolynomialRidgeFunction(LegendreTensorBasis(n,p), coef, U)
	
	X = np.random.randn(M,m)
	fX = prf.eval(X) 
	return X, fX

def test_fit_inf():
	X, fX = exact_data()

	pra = PolynomialRidgeApproximation(degree = 3, subspace_dimension = 1, norm = np.inf)
	pra.fit(X, fX)
	assert np.all(np.isclose(pra(X), fX))

def test_fit_one():
	X, fX = exact_data()

	pra = PolynomialRidgeApproximation(degree = 3, subspace_dimension = 1, norm = 1)
	pra.fit(X, fX)
	assert np.all(np.isclose(pra(X), fX))
	
