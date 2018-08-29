import numpy as np
from psdr.ridge import residual, build_J, LegendreTensorBasis  

def check_gradient(f, x, grad, verbose = False):
	grad_est_best = np.zeros(grad.shape)
	for h in np.logspace(-14, -2,10):
		grad_est = np.zeros(grad.shape)
		for i in np.ndindex(x.shape):
			# Build unit vector
			ei = np.zeros(x.shape)
			ei[i] = 1
			# Construct finite difference approximation
			d = (f(x + h*ei) - f(x - h*ei))/(2*h)
			# Add to gradient approximation
			idx = tuple([slice(0,ni) for ni in d.shape] + list(i))
			grad_est[idx] = (f(x + h*ei) - f(x - h*ei))/(2*h)
		if np.max(np.abs(grad - grad_est)) < np.max(np.abs(grad - grad_est_best)):
			grad_est_best = grad_est

	if verbose:
		for i in np.ndindex(grad.shape):
			print i, "%+5.2e %+5.2e  %+5.2e" %(grad[i], grad_est_best[i], grad[i]/grad_est_best[i])

	return np.max(np.abs(grad - grad_est_best))


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
