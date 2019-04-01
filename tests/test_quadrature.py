from __future__ import print_function

import numpy as np
import psdr
from psdr.quadrature import gauss
import scipy.integrate

def test_gauss():
	func = lambda x: np.sin(5*x+0.1) + x**2
	
	x, w = gauss(10, -1,1)
	int1 = np.sum( w*func(x))
	
	int2, err = scipy.integrate.quadrature(func, -1, 1)
	print("gauss", int1)
	print("scipy", int2)

	assert np.isclose(int1, int2), "Quadrature rule failed"

def test_quad():
	# test tensor-product quadrature rule

	func = lambda x: np.sin(5*x[0]+0.1) + (x[1]+0.1)**2

	dom = psdr.BoxDomain([-1,-1], [1,1])
	X, w = dom.quadrature_rule(100, method = 'gauss')
	int1 = np.sum([wi*func(xi) for xi, wi in zip(X, w)])
	

	int2, err = scipy.integrate.nquad(lambda x1, x2: func([x1,x2]), [ (lbi, ubi) for lbi, ubi in zip(dom.lb, dom.ub)])

	print("gauss", int1)
	print("scipy", int2)

	assert np.isclose(int1, int2), "Quadrature rule failed"
	
