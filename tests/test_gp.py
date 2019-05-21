from __future__ import print_function
import numpy as np
from psdr import GaussianProcess, BoxDomain, Function 
from checkder import check_gradient 

def test_gp_der(m = 5, M = 50):
	np.random.seed(0)
	X = np.random.uniform(-1,1, size = (M, m))
	a = np.ones(m)
	y = np.dot(a.T, X.T).T + 1
	
	
	A = np.random.randn(m,m)
	Q, R = np.linalg.qr(A)
	Lfixed = R.T
	L0 = 10e-1*Lfixed

	tol = 1e-4 
	
	ell0 =  np.array([1.])
	print("constant")
	for degree in [None, 0, 1, 2]:
		gp = GaussianProcess(structure = 'const', degree = degree)
		gp._fit_init(X, y)
		obj = lambda ell: gp._obj(ell, X, y)		
		grad = lambda ell: gp._grad(ell, X, y)		

		assert check_gradient(ell0, obj, grad) < tol
	
	print("scalar_mult")
	for poly_degree in [None, 0, 1, 2]:
		gp = GaussianProcess(structure = 'scalar_mult', degree = degree, Lfixed = L0)
		gp._fit_init(X, y)
		obj = lambda ell: gp._obj(ell, X, y)		
		grad = lambda ell: gp._grad(ell, X, y)		

		assert check_gradient(ell0, obj, grad) < tol

	ell0 = np.ones(m)
	print("diag")
	for poly_degree in [None, 0, 1, 2]:
		gp = GaussianProcess(structure = 'diag', degree = degree)
		gp._fit_init(X, y)
		obj = lambda ell: gp._obj(ell, X, y)		
		grad = lambda ell: gp._grad(ell, X, y)

		gp._fit_init(X, y)
		assert check_gradient(ell0, obj, grad) < tol

	print("tril")
	ell0 = np.array([ L0[i,j] for i, j in zip(*np.tril_indices(m))])
	for degree in [None, 0, 1, 2]:
		gp = GaussianProcess(structure = 'tril', degree = degree)
		gp._fit_init(X, y)
		obj = lambda ell: gp._obj(ell, X, y)		
		grad = lambda ell: gp._grad(ell, X, y)

		assert check_gradient(ell0, obj, grad) < tol


def test_gp_fit(m = 3, M = 100):
	""" check """
	dom = BoxDomain(-np.ones(m), np.ones(m))
	a = np.ones(m)
	b = np.ones(m)
	b[0] = 0
	f = lambda x: np.sin(x.dot(a)) + x.dot(b)**2
	fun = Function(f, dom) 
	X = dom.sample(M)
	fX = f(X)
	for structure in ['const', 'diag', 'tril']:
		for degree in [None, 0, 1]:
			gp = GaussianProcess(structure = structure, degree = degree)
			gp.fit(X, fX)
			print(gp.L)
			I = ~np.isclose(gp(X), fX)
			print(fX[I])
			print(gp(X[I]))
			assert np.all(np.isclose(gp(X), fX, atol = 1e-5)), "we should interpolate samples"

			_, cov = gp.eval(X, return_cov = True)
			assert np.all(np.isclose(cov, 0, atol = 1e-3)), "Covariance should be small at samples"

if __name__ == '__main__':
	test_gp_fit()


