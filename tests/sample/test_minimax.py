from __future__ import print_function
import numpy as np
from scipy.spatial.distance import cdist
import cvxpy as cp

import psdr
from psdr.sample.minimax import _cq_center_cvxpy


def test_cq_center(m = 3, q = 10):
	np.random.seed(1)
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	X = dom.sample(10)
	
	# Isotropic case
	I = np.eye(len(dom))
	L1 = np.random.randn(2,m)
	L2 = np.random.randn(m,m)
	for L in [I, L1, L2]:
		Y = L.dot(X.T).T
		xhat1 = _cq_center_cvxpy(Y, L, q = q)

		# Naive solve
		xhat = cp.Variable(m)
		obj = cp.sum([cp.norm(xhat.__rmatmul__(L) - y)**q for y in Y])
		prob = cp.Problem(cp.Minimize(obj))
		prob.solve()
		xhat2 = xhat.value
		print(xhat1)
		print(xhat2)
		
		print("mismatch", np.linalg.norm(L.dot(xhat1 - xhat2)))
		assert np.linalg.norm(L.dot(xhat1 - xhat2)) < 1e-5


def test_minimax(m = 3, N = 5):
	np.random.seed(1)
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	Xhat = psdr.minimax_cluster(dom, N, N0 = 1e3)

	# Simply see if these points are better than random points
	X = dom.sample_grid(10)
	def minimax_score(Xhat):
		return np.max(np.min(cdist(Xhat, X), axis = 0))

	Xhat2 = dom.sample(N)	
	print("score minimax", minimax_score(Xhat))
	print("score random ", minimax_score(Xhat2)) 	
	assert minimax_score(Xhat) <= minimax_score(Xhat2)


def test_minimax_conditioning(m = 5, N = 200):
	np.random.seed(0)
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))

	psdr.minimax_cluster(dom, N, verbose = True, maxiter =5)

#def test_minimax_parallel(m = 5, N = 10):
#	import time
#
#	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
#
#	np.random.seed(0)
#	t0 = time.time()
#	X1 = psdr.minimax_cluster(dom, N, verbose = True, parallel = True, maxiter = 5)
#	t1 = time.time()
#	print("\n\t time parallel  ", t1-t0)
#	t0 = time.time()
#	X2 = psdr.minimax_cluster(dom, N, verbose = True, parallel = False, maxiter = 5)
#	t1 = time.time()
#	print("\n\t time sequential", t1-t0)
#	
#	assert np.all(np.isclose(X1, X2))

if __name__ == '__main__':
	#test_cq_center()
	#test_minimax()
	test_minimax_conditioning()
#	test_minimax_parallel()
