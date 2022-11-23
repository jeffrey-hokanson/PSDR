from __future__ import print_function
import numpy as np
import scipy.linalg
from psdr import PolynomialRidgeApproximation, LegendreTensorBasis, PolynomialRidgeFunction, ArnoldiPolynomialBasis
from checkder import *
from itertools import product

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
	coef = np.random.randn(len(LegendreTensorBasis(p, dim = n)))
	prf = PolynomialRidgeFunction(LegendreTensorBasis(p, dim = n), coef, U)

	x = np.random.randn(m)

	print(prf.eval(x))
	print(prf.grad(x))
	print(prf.hessian(x))
	
	assert check_derivative(x, prf.eval, lambda x: prf.grad(x) ) < 1e-7
	assert check_hessian(x, prf.eval, lambda x: prf.hessian(x) ) < 1e-5
		
	

def test_varpro_jacobian():
	np.random.seed(1)
	M = 6
	m = 2
	n = 1
	p = 2
	
	# Samples
	X = np.random.uniform(-1,1, size = (M,m))
	
	# Synthetic function
	a = np.random.randn(m)
	b = np.random.randn(m)
	fX = np.dot(a,X.T)**2 + np.dot(b, X.T)**3

	# Random point
	U, _ = np.linalg.qr(np.random.randn(m,n))

	U_flat = U.flatten()

	for basis in ['legendre', 'arnoldi']:
		pra = PolynomialRidgeApproximation(degree = p, subspace_dimension = n, scale = False, basis = basis)
		# This sets the basis
		Y = (U.T @ X.T).T
	#	pra.basis = pra.Basis(pra.degree, Y) 
		#pra._varpro_jacobian(X, fX, U)
		res = lambda U: pra._varpro_residual(X, fX, U)
		jac = lambda U: pra._varpro_jacobian(X, fX, U)

		err = check_jacobian(U_flat, res, jac, hvec = [1e-7])	

		assert err < 1e-6
	

	# Check with scaling on
	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension = n, scale = True)
	#pra.set_scale(X, U)
	res = lambda U: pra._varpro_residual(X, fX, U)
	jac = lambda U: pra._varpro_jacobian(X, fX, U)

	err = check_jacobian(U_flat, res, jac)	
	assert err < 1e-6
	

	# Check with scaling on for Hermite basis
	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension = n, scale = True, basis = 'hermite')
#	pra.set_scale(X, U)
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

	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension = n, scale = False, maxiter = 0)
	
	# Initialize subspace
	pra.fit(X, fX)
	#pra.set_scale(X, U)
	#pra._fit_fixed_U_inf_norm(X, fX, U)
	#c = pra.coef	
	c = np.random.randn(len(pra.Basis(p, dim = n)))	

	U_c = np.hstack([U.flatten(), c])

	res = lambda U_c: pra._residual(X, fX, U_c)
	jac = lambda U_c: pra._jacobian(X, fX, U_c)
	
	print(res(U_c))
	print(jac(U_c))

	err = check_jacobian(U_c, res, jac)	
	assert err < 1e-6

def test_exact():
	np.random.seed(1)
#	M = 100
#	m = 10
#	n = 2
#	p = 5
#	
#	# Samples
#	X = np.random.randn(M,m)
#	
#	# Synthetic function
#	a = np.random.randn(m)
#	b = np.random.randn(m)
#	fX = np.dot(a,X.T)**2 + np.dot(b, X.T)**3
	M = 1000
	m, n = 10, 2
	p = 3
	X, fX, Uopt = exact_data(M = M, m = m, n = n, p =p)

	# Random point
	U, _ = np.linalg.qr(np.random.randn(m,n))
	# Actual ridge subspace
	#U, _ = np.linalg.qr(np.vstack([a,b]).T)	
	#U += Uopt

	for basis in ['legendre', 'arnoldi']:

		pra = PolynomialRidgeApproximation(degree = p, subspace_dimension = n, scale = True, basis = basis, verbose = True)
		pra.fit(X, fX, U0 = U)
		# Because the data is an exact ridge function, we should (I think) converge to the global solution
		for fX1, fX2 in zip(pra(X), fX):
			print("%+10.5e  %+10.5e | %10.5e" % (fX1,fX2, np.abs(fX1 - fX2)))
		assert np.all(np.isclose(pra(X), fX))

def exact_data(M = 100, m = 10, n = 1, p = 3):
	U = scipy.linalg.orth(np.random.randn(m,n))
	fX = np.random.randn(M,)
	X = np.random.rand(M, m)
	Y = (U.T @ X.T).T
	basis = ArnoldiPolynomialBasis(p, Y)
	V = basis.V()
	
	# Project to be exact
	fX = V @ (V.T @ fX)
	return X, fX, U

def test_fit_inf():
	X, fX, Uopt = exact_data()

	pra = PolynomialRidgeApproximation(degree = 3, subspace_dimension = 1, norm = np.inf, verbose = True)
	pra.fit(X, fX)
	assert np.all(np.isclose(pra(X), fX))

def test_fit_one():
	X, fX, Uopt = exact_data()

	pra = PolynomialRidgeApproximation(degree = 3, subspace_dimension = 1, norm = 1, verbose = True)
	pra.fit(X, fX)
	assert np.all(np.isclose(pra(X), fX))
	
def test_fit_two():
	X, fX, Uopt = exact_data()

	pra = PolynomialRidgeApproximation(degree = 3, subspace_dimension = 1, norm = 2, verbose = True)
	pra.fit(X, fX)
	assert np.all(np.isclose(pra(X), fX))


def test_profile(degree = 3, subspace_dimension = 1):
	X, fX, Uopt = exact_data()
	pra = PolynomialRidgeApproximation(degree = 3, subspace_dimension = 1, norm = 2, verbose = True)
	pra.fit(X, fX)
	
	Y = pra.U.T.dot(X.T).T
	assert np.all(np.isclose(pra.profile(Y), pra(X)))


def test_same_solution(degree = 5, dim = 2):
	np.random.seed(0)
	X, fX, U = exact_data(M = 1000, m = 10, n = dim, p = degree)

	bases = ['arnoldi', 'legendre', 'monomial', 'chebyshev', 'laguerre', 'hermite']	

	#
	#U0 = U + 0.01*np.random.randn(*U.shape)
	U0 = None
	for norm, bound in product([1, 2, np.inf], [None]): # , 'lower', 'upper']):
		print("="*50)
		print("Norm", norm, "bound" ,bound)
		Us = []
		for basis in bases:
			print('basis', basis)
			pra = PolynomialRidgeApproximation(degree = 5, subspace_dimension = dim, basis = basis, bound = bound, verbose = True)
			pra.fit(X, fX, U0 = U0)
			Us.append(np.copy(pra.U))

		angles = np.zeros((len(Us), len(Us)))
		for (i, j) in zip(*np.triu_indices(len(Us))):
			angles[i,j] = np.max(scipy.linalg.subspace_angles(Us[i], Us[j]))

		print(angles)
		assert np.max(angles) < 1e-10, "Did not find same solution using a different basis"

if __name__ == '__main__':
#	test_exact()
#	test_varpro_jacobian()
	test_same_solution()
