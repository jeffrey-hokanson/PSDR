from __future__ import print_function

import numpy as np
from checkder import check_derivative
from psdr.demos import AffineFunction, QuadraticFunction

def test_grad():
	aff = AffineFunction()
	quad = QuadraticFunction()
	quad2 = QuadraticFunction(linear = np.ones(5), constant = 5.)


	for fun in [aff, quad, quad2]:
		print(fun)
		x = fun.domain.sample(1)
		grad = lambda x: fun.grad(x).T
		assert check_derivative(x, fun, grad) < 1e-7*np.linalg.norm(grad(x), np.inf)	

