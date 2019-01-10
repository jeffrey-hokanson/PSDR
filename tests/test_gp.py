import numpy as np
from psdr import GaussianProcess 
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
	for degree in [None, 0, 1, 2]:
		gp = GaussianProcess(structure = 'const', degree = degree)
		gp._fit_init(X, y)
		obj = lambda ell: gp._obj(ell, X, y)		
		grad = lambda ell: gp._grad(ell, X, y)		

		assert check_gradient(ell0, obj, grad) < tol
	
	for poly_degree in [None, 0, 1, 2]:
		gp = GaussianProcess(structure = 'scalar_mult', degree = degree, Lfixed = L0)
		gp._fit_init(X, y)
		obj = lambda ell: gp._obj(ell, X, y)		
		grad = lambda ell: gp._grad(ell, X, y)		

		assert check_gradient(ell0, obj, grad) < tol

	ell0 = np.ones(m)
	for poly_degree in [None, 0, 1, 2]:
		gp = GaussianProcess(structure = 'diag', degree = degree)
		gp._fit_init(X, y)
		obj = lambda ell: gp._obj(ell, X, y)		
		grad = lambda ell: gp._grad(ell, X, y)

		gp._fit_init(X, y)
		assert check_gradient(ell0, obj, grad) < tol

	ell0 = np.array([ L0[i,j] for i, j in zip(*np.tril_indices(m))])
	for poly_degree in [None, 0, 1, 2]:
		gp = GaussianProcess(structure = 'tril', degree = degree)
		gp._fit_init(X, y)
		obj = lambda ell: gp._obj(ell, X, y)		
		grad = lambda ell: gp._grad(ell, X, y)

		assert check_gradient(ell0, obj, grad) < tol


# TODO: Check solution vs. sklearn

if __name__ == '__main__':
	test_gp_der()


