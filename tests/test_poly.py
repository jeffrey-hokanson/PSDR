from __future__ import print_function
import numpy as np

from psdr import PolynomialApproximation, PolynomialFunction, LegendreTensorBasis, BoxDomain
from checkder import *

def test_poly_der(dimension = 3, degree = 5):
	np.random.seed(0)
	coef = np.random.randn(len(LegendreTensorBasis(dimension, degree)))
	pf = PolynomialFunction(dimension, degree, coef)

	x = np.random.randn(dimension)
	print(x)
	print(pf.eval(x))
	print(pf.grad(x))

	assert check_derivative(x, pf.eval, lambda x: pf.grad(x).flatten() ) < 1e-7

def test_poly_hess(dimension = 3, degree = 5):
	np.random.seed(0)
	coef = np.random.randn(len(LegendreTensorBasis(dimension, degree)))
	pf = PolynomialFunction(dimension, degree, coef)
	
	x = np.random.randn(dimension)
	print(x)
	print(pf.eval(x))
	print(pf.hessian(x))

	assert check_hessian(x, pf.eval, lambda x: pf.hessian(x).reshape(dimension, dimension) ) < 5e-5

def test_dimensions(dimension = 3, degree = 5):
	np.random.seed(0)
	coef = np.random.randn(len(LegendreTensorBasis(dimension, degree)))
	pf = PolynomialFunction(dimension, degree, coef)

	x = np.random.randn(dimension)
	X = np.random.randn(10, dimension)
	
	print('pf.eval(x).shape', pf.eval(x).shape) 
	assert pf.eval(x).shape == (1,)
	print('pf.eval(X).shape', pf.eval(X).shape) 
	assert pf.eval(X).shape == (10,)

	# gradients
	print('pf.grad(x).shape', pf.grad(x).shape) 
	assert pf.grad(x).shape == (dimension,)
	print('pf.grad(X).shape', pf.grad(X).shape) 
	assert pf.grad(X).shape == (10,dimension)
	
	# Hessians
	print('pf.hessian(x).shape', pf.hessian(x).shape) 
	assert pf.hessian(x).shape == (dimension,dimension,)
	print('pf.hessian(X).shape', pf.hessian(X).shape) 
	assert pf.hessian(X).shape == (10,dimension, dimension)
	

def test_poly_basis(dimension = 2, degree = 5):
	""" test different bases"""	
	np.random.seed(0)
	coef = np.random.randn(len(LegendreTensorBasis(dimension, degree)))
	pf = PolynomialFunction(dimension, degree, coef)

	dom = BoxDomain(-np.ones(dimension), np.ones(dimension))
	X = dom.sample(100)
	fX = pf(X)
	Xtest = dom.sample(1000)
	fXtest = pf(Xtest)
	
	for basis in ['legendre', 'monomial', 'chebyshev', 'laguerre', 'hermite']:
		pa = PolynomialApproximation(degree, basis = basis)
		pa.fit(X, fX)
		assert np.linalg.norm(pa(Xtest) - fXtest, np.inf) < 1e-7


def test_poly_fit(dimension = 2, degree = 5, tol = 1e-6):
	np.random.seed(0)
	dom = BoxDomain(-np.ones(dimension), np.ones(dimension))
	X = dom.sample(100)

	fXnoise = np.random.randn(X.shape[0])

	for bound in ['lower', 'upper', None]:
		for norm in [1,2, np.inf]:
			for basis in ['legendre', 'monomial', 'chebyshev', 'laguerre', 'hermite']:
				pa = PolynomialApproximation(degree, basis = basis, norm = norm, bound = bound)
				pa.fit(X, fXnoise)
				if bound == 'lower':
					I = ~(pa(X) - tol <= fXnoise)
					if np.sum(I) > 0:
						print('%s lower:' % basis)
						print(pa(X[I]), fXnoise[I])
						assert False
				if bound == 'upper':
					I = ~(pa(X) + tol >= fXnoise)
					if np.sum(I) > 0:
						print('%s upper:' % basis)
						print(pa(X[I]), fXnoise[I])
						assert False
					#assert np.all(pa(X) >= fXnoise -1e-7)


def test_roots(degree =  5):
	X = np.random.randn(degree+1,1)
	fX = np.random.randn(degree+1)
	for basis in ['legendre', 'monomial', 'chebyshev', 'laguerre', 'hermite']:
		print("basis", basis)
		pa = PolynomialApproximation(degree, basis = basis)
		pa.fit(X, fX)
		assert np.all(np.isclose(pa(X), fX)), "Approximation failed"
		r = pa.roots()
		for rr in r:
			print(rr, pa(rr))
			assert np.isclose(pa(rr), 0)

if __name__ == '__main__':
	test_dimensions()	
