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

if __name__ == '__main__':
	test_func()	
