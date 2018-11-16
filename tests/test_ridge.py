import numpy as np
from psdr.poly_ridge import residual, build_J, LegendreTensorBasis  
from psdr.util import check_gradient


def test_jacobian():
	np.random.seed(1)
	M = 100
	m = 10
	n = 2
	p = 5
	X = np.random.uniform(-1,1, size = (M,m))
	a = np.random.randn(m)
	b = np.random.randn(m)
	fX = np.dot(a,X.T)**2 + np.dot(b, X.T)**3
	U, _ = np.linalg.qr(np.random.randn(m,n))

	basis = LegendreTensorBasis(n,p)
	J = build_J(U, X, fX, basis)
	res = lambda U: residual(U, X, fX, basis)
	err = check_gradient(res, U, J)
	assert err < 1e-6

def test_DV():
	M = 10
	n = 2
	p = 3
	Y = np.random.uniform(-1,1, size = (M,n))
		
	basis = LegendreTensorBasis(n,p)
	V = basis.V(Y)
	N = V.shape[1]
	DV = basis.DV(Y)
	DVtrue = np.zeros((M, N, M, n))
	for k in range(M):
		DVtrue[k,:,k,:] = DV[k,:,:]
	
	err = check_gradient(lambda Y: basis.V(Y), Y, DVtrue, verbose = False)
	assert err < 1e-7

if __name__ == '__main__':
	#test_jacobian()
	test_DV()
