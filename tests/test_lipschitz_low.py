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


def test_lipschitz_low(N = 50, M = 15):
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
		
	lip = psdr.LipschitzMatrix()
	lip1 = psdr.LowRankLipschitzMatrix(1, verbose = True)
	lip2 = psdr.LowRankLipschitzMatrix(2, verbose = True)
	lip5 = psdr.LowRankLipschitzMatrix(len(fun.domain)-1, verbose = True)

	for lip in [ ]:
		for kwargs in [ {'X': X, 'fX': fX}, {'grads':grads}, {'X':X, 'fX': fX, 'grads': grads}]:
			lip.fit(**kwargs)
			err = check_lipschitz(lip.H, **kwargs) 
			print("error", err)
			assert err > -1e-6, "constraints not satisfied"
	
	# If we fit an m-1 dimensional case we should get back the true Lipschitz matrix
	# We avoid the low rank problem of points by considering gradients here 
	lip5.fit(grads = grads)
	lip.fit(grads = grads)
	print(np.linalg.eigvalsh(lip5.H))
	print(np.linalg.eigvalsh(lip.H))
	print("True Lipschitz")
	print(lip.H)
	print("Low Rank Lipschitz")
	print(lip5.H)
	print("J")
	print(lip5.J)
	print("U")
	print(lip5.U)
	print("alpha", lip5.alpha)


if __name__ == '__main__':
	test_lipschitz_low()
 

