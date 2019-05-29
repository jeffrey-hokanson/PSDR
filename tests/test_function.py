from __future__ import print_function
import numpy as np
from psdr import BoxDomain, Function

from dask.distributed import Client
from checkder import *  


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
		return a.dot(X.T)
	
	def fun_b(X):
		return b.dot(X.T)
	
	def fun(X):
		return np.vstack([X.dot(a), X.dot(b)]).T


	print("Single function with multiple outputs")
	for vectorized in [True, False]:
		myfun = Function(fun, dom, vectorized = vectorized)
		print(fun(X))
		print("vectorized", vectorized)
		print(myfun(X).shape)
		assert myfun(X).shape == (M, 2) 
		print(myfun(X[0]).shape)
		assert myfun(X[0]).shape == (2,)

		fX = fun(X)
		for i, x in enumerate(X):
			assert np.all(np.isclose(fX[i], fun(x)))
	
	print("Two functions with a single output each")
	for vectorized in [True, False]:
		myfun = Function([fun_a, fun_b], dom, vectorized = vectorized)
		print(fun(X))
		print("vectorized", vectorized)
		print(myfun(X).shape)
		assert myfun(X).shape == (M, 2) 
		print(myfun(X[0]).shape)
		assert myfun(X[0]).shape == (2,)
		
		fX = fun(X)
		for i, x in enumerate(X):
			assert np.all(np.isclose(fX[i], fun(x)))


def test_finite_diff():
	from psdr.demos.golinski import golinski_volume, build_golinski_design_domain
	
	dom = build_golinski_design_domain()
	fun = Function(golinski_volume, dom, fd_grad = True)

	x = fun.domain.sample()
	print(x)
	print(fun(x))
	print(fun.grad(x))	
	err = check_derivative(x, fun.eval, fun.grad)
	print(err)
	assert err < 1e-5

	# Check asking for multiple gradients
	X = fun.domain.sample(5)
	grads = fun.grad(X)
	for i, x in enumerate(X):
		assert np.all(np.isclose(grads[i], fun.grad(x)))
	

def test_return_grad(m=3):
	A = np.random.randn(m, m)
	A += A.T

	def func(x, return_grad = False):
		fx = 0.5*x.dot(A.dot(x))

		if return_grad:
			grad = A.dot(x)
			return fx, grad
		else:
			return fx
	
	dom = BoxDomain(-2*np.ones(m), 2*np.ones(m))
	x = dom.sample(1)

	fun = Function(func, dom)	
	# Check the derivative
	x_norm = dom.normalize(x)
	err = check_derivative(x_norm, fun.eval, fun.grad)
	assert err < 1e-5

	# Check wrapping
	fx, grad = func(x, return_grad = True)
	assert np.isclose(fx, fun(x_norm))
	# multiply the grad by two to correct the change of coordinates
	assert np.all(np.isclose(2*grad, fun.grad(x_norm)))

	# Check multiple outputs
	X = dom.sample(10)
	fX, grads = fun(X, return_grad = True)
	for i, x in enumerate(X):
		assert np.isclose(fun(x), fX[i])
		assert np.all(np.isclose(fun.grad(x), grads[i]))

	# Check vectorized functions
	def func2(X, return_grad = False):
		X = np.atleast_2d(X)
		fX = np.vstack([0.5*x.dot(A.dot(x)) for x in X])
		if return_grad:
			grad = X.dot(A)
			return fX, grad
		else:
			return fX

	fun2 = Function(func2, dom, vectorized = True)

	x = fun2.domain.sample()
	X = fun2.domain.sample(5)
	assert np.isclose(fun2(x), fun(x)) 
	assert np.all(np.isclose(fun2(X), fun(X)))
	assert np.all(np.isclose(fun2.grad(X), fun.grad(X)))
	
if __name__ == '__main__':
	#test_mult_output()
	#test_finite_diff()	
	test_return_grad()
