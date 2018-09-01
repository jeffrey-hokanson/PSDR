import numpy as np
from psdr import fit_gp


def test_gp_der(m = 3, M = 50):
	X = np.random.uniform(-1,1, size = (M, m))
	a = np.ones(m)
	y = np.dot(a.T, X.T).T**2 + 1
	
	
	A = np.random.randn(m,m)
	Q, R = np.linalg.qr(A)
	Lfixed = R.T
	L0 = 5e-1*Lfixed

	tol = 1e-5 
		
#	assert fit_gp(X, y, structure = 'const', L0 = L0, _check_gradient = True) < tol
#	assert fit_gp(X, y, structure = 'const', L0 = L0,poly_degree = 0, _check_gradient = True) < tol
#	assert fit_gp(X, y, structure = 'const', L0 = L0,poly_degree = 1, _check_gradient = True) < tol
#	assert fit_gp(X, y, structure = 'const', L0 = L0,poly_degree = 2, _check_gradient = True) < tol
#
#	assert fit_gp(X, y, structure = 'scalar_mult', Lfixed = Lfixed, _check_gradient = True) < tol
#	assert fit_gp(X, y, structure = 'scalar_mult', Lfixed = Lfixed, poly_degree = 0, _check_gradient = True) < tol
#	assert fit_gp(X, y, structure = 'scalar_mult', Lfixed = Lfixed, poly_degree = 1, _check_gradient = True) < tol
#	assert fit_gp(X, y, structure = 'scalar_mult', Lfixed = Lfixed, poly_degree = 2, _check_gradient = True) < tol
#
#	assert fit_gp(X, y, structure = 'diag', L0 = L0, _check_gradient = True) < tol
#	assert fit_gp(X, y, structure = 'diag', L0 = L0, poly_degree = 0, _check_gradient = True) < tol
#	assert fit_gp(X, y, structure = 'diag', L0 = L0, poly_degree = 1, _check_gradient = True) < tol
#	assert fit_gp(X, y, structure = 'diag', L0 = L0, poly_degree = 2, _check_gradient = True) < tol
#
	assert fit_gp(X, y, structure = 'tril', L0 = L0, _check_gradient = True) < tol
	assert fit_gp(X, y, structure = 'tril', L0 = L0, poly_degree = 0, _check_gradient = True) < tol
	assert fit_gp(X, y, structure = 'tril', L0 = L0, poly_degree = 1, _check_gradient = True) < tol
	assert fit_gp(X, y, structure = 'tril', L0 = L0, poly_degree = 2, _check_gradient = True) < tol
	
#	assert fit_gp(X, y, structure = 'tril', L0 = L0, _check_gradient = True, rank = 1) < tol
#	assert fit_gp(X, y, structure = 'tril', L0 = L0, _check_gradient = True, rank = 2) < tol
#	assert fit_gp(X, y, structure = 'tril', L0 = L0, _check_gradient = True, rank = 3) < tol
	print  fit_gp(X, y, structure = 'tril', L0 = L0, _check_gradient = True, rank = None, poly_degree = None)
#	assert fit_gp(X, y, structure = 'tril', L0 = L0, _check_gradient = True, rank = 3, poly_degree = None) < tol


# TODO: Check solution vs. sklearn


if __name__ == '__main__':
	test_gp_der()


