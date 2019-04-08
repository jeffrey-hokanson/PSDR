from __future__ import print_function
import numpy as np
from psdr import BoxDomain, Function

from dask.distributed import Client



def test_lambda():
	dom = BoxDomain(-1,1)
	def f(x):
		return x	
	#f = lambda x: x
	print('about to start client')
	
	# We use a threaded version for sanity
	# https://github.com/dask/distributed/issues/2515
	client = Client(processes = False)
	print(client)
	fun = Function(f, dom, dask_client = client)

	x = dom.sample(1)
	res = fun.eval_async(x)
	print(x, res.result())
	assert np.isclose(x, res.result())

def func(x):
	return x

def test_func():
	client = Client(processes = False)
	dom = BoxDomain(-1,1)
	fun = Function(func, dom, dask_client = client)
	X = dom.sample(5)
	res = fun.eval_async(X)
	for r, x in zip(res, X):
		print(r.result())
		assert np.isclose(x, r.result())


def test_mult_output(M= 10,m = 5):
	dom = BoxDomain(-np.ones(m), np.ones(m))
	X = dom.sample(M)

	a = np.random.randn(m)
	b = np.random.randn(m)

	def fun_a(X):
		return X.T.dot(a)
	
	def fun_b(X):
		return X.T.dot(b)
	
	def fun(X):
		return np.vstack([X.dot(a), X.dot(b)]).T


	for vectorized in [True, False]:
		myfun = Function(fun, dom, vectorized = vectorized)
		print(fun(X))
		print("vectorized", vectorized)
		print(myfun(X).shape)
		assert myfun(X).shape == (M, 2) 
		print(myfun(X[0]).shape)
		assert myfun(X[0]).shape == (2,)

		
if __name__ == '__main__':
	test_mult_output()	
