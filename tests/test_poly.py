from __future__ import print_function
import numpy as np

from psdr import PolynomialApproximation, PolynomialFunction, LegendreTensorBasis
from checkder import *

def test_poly_der(dimension = 3, degree = 5):
	coef = np.random.randn(len(LegendreTensorBasis(dimension, degree)))
	pf = PolynomialFunction(dimension, degree, coef)

	x = np.random.randn(dimension)
	print(x)
	print(pf.eval(x))
	print(pf.grad(x))

	assert check_derivative(x, pf.eval, lambda x: pf.grad(x).flatten() ) < 1e-7

def test_poly_hess(dimension = 3, degree = 5):
	coef = np.random.randn(len(LegendreTensorBasis(dimension, degree)))
	pf = PolynomialFunction(dimension, degree, coef)
	
	x = np.random.randn(dimension)
	print(x)
	print(pf.eval(x))
	print(pf.hessian(x))

	assert check_hessian(x, pf.eval, lambda x: pf.hessian(x).reshape(dimension, dimension) ) < 1e-5
