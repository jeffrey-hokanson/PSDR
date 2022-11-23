from __future__ import print_function, division

import numpy as np

import psdr, psdr.demos


def check_lipschitz(H, X = None, fX = None, grads = None):
	err = np.inf
	if X is not None and fX is not None:
		for i in range(len(X)):
			for j in range(i+1,len(X)):
				y = X[i] - X[j]
				gap = y.dot(H.dot(y)) - (fX[i] - fX[j])**2
				#print("samp %3d %3d : %8.3e" % (i,j, gap))
				err = min(gap, err)

	if grads is not None:
		for i, g in enumerate(grads):
			gap = np.min(np.linalg.eigvalsh(H - np.outer(g,g)))
			err = min(float(gap), err)
			#print("grad %3d  : %8.3e" % (i, gap))

	return err


def test_lipschitz_fixed_U(N = 10, M = 20):
	np.random.seed(0)
	if True:
		fun = psdr.demos.HartmannMHD()
		X = fun.domain.sample(M)
		fX = fun(X)[:,0]
		Xg = fun.domain.sample(N)
		grads = fun.grad(Xg)[:,0,:]
	else:
		fun = psdr.demos.OTLCircuit()
		X = fun.domain.sample(M)
		fX = fun(X)
		Xg = fun.domain.sample(N)
		grads = fun.grad(Xg)

	m = len(fun.domain)
	#grads = np.zeros((0,m))
	X = np.zeros((0,m))
	fX = np.zeros((0,))

	lipr = psdr.PartialLipschitzMatrix(m)
	U = np.eye(m)
	J, alpha = lipr._fixed_U(U, X, fX, grads, 0)
	
	lip = psdr.LipschitzMatrix(method = 'cvxpy')
	lip.fit(X = X, fX = fX, grads = grads)

	assert np.max(np.abs(lip.H - J)) < 1e-3

def test_lipschitz_partial(N = 30, M = 20):
	np.random.seed(0)
	if True:
		fun = psdr.demos.HartmannMHD()
		X = fun.domain.sample(M)
		fX = fun(X)[:,0]
		Xg = fun.domain.sample(N)
		grads = fun.grad(Xg)[:,0,:]
	else:
		fun = psdr.demos.OTLCircuit()
		X = fun.domain.sample(M)
		fX = fun(X)
		Xg = fun.domain.sample(N)
		grads = fun.grad(Xg)
		
	lip1 = psdr.PartialLipschitzMatrix(1, verbose = True, maxiter = 10)
	lip2 = psdr.PartialLipschitzMatrix(2, verbose = True, maxiter = 10)

	for lip in [lip1, lip2 ]:
		for kwargs in [ {'X': X, 'fX': fX}, {'grads':grads}, {'X':X, 'fX': fX, 'grads': grads}]:
			lip.fit(**kwargs)
			err = check_lipschitz(lip.H, **kwargs) 
			print("error", err)
			assert err > -1e-6, "constraints not satisfied"

			# Check square-root L
			err_L = np.max(np.abs(lip.H - lip.L.dot(lip.L))) 
			print("err L", err_L)
			assert err_L < 1e-7
	
	# If we fit an m-1 dimensional case we should get back the true Lipschitz matrix
	# We avoid the low rank problem of points by considering gradients here 
	lip = psdr.LipschitzMatrix()
	lip.fit(grads = grads)
	H = lip.H

	lip3 = psdr.PartialLipschitzMatrix(len(fun.domain)-1, verbose = True, U0 = lip.U[:,0:len(fun.domain)-1])
	lip4 = psdr.PartialLipschitzMatrix(len(fun.domain), verbose = True)
	for lip in [lip3, lip4]:
		lip.fit(grads = grads)
		err = np.max(np.abs(H - lip.H))
		print('error', err)
		assert err < 1e-4, "Did not identify true Lipschitz matrix"


if __name__ == '__main__':
	test_lipschitz_partial()
	#test_lipschitz_fixed_U()
 

