from __future__ import print_function
import numpy as np
import psdr

def test_fit_function(m = 4):
	A = np.random.randn(m,2)
	A = A.dot(A.T)

	f = lambda x: 0.5*float(x.dot(A.dot(x)))
	grad = lambda x: A.dot(x)
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))

	fun = psdr.Function(f, dom, grads = grad)
	X = fun.domain.sample(10)
	fX = fun(X)
	fun.grad(X)

	act = psdr.ActiveSubspace()
	act.fit_function(fun, 1e3)
	print(act.singvals)
	assert np.sum(~np.isclose(act.singvals,0)) == 2

if __name__ == '__main__':
	test_fit_function()
