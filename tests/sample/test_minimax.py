import numpy as np

from scipy.spatial.distance import cdist

import psdr
from psdr.sample.minimax import _cq_center_cvxpy, _cq_center_agd

def test_cq_center(m = 3):
	np.random.seed(1)
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	X = dom.sample(10)
	
	# Isotropic case
	I = np.eye(len(dom))
	L1 = np.random.randn(2,m)
	L2 = np.random.randn(m,m)
	for L in [I, L1, L2]:
		Y = L.dot(X.T).T
		xhat1 = _cq_center_cvxpy(Y, L, q = 10)
		xhat2 = _cq_center_agd(X, L, q = 10, verbose = False, xtol = 1e-10, maxiter = int(1e4))

		print(xhat1)
		print(xhat2)
		
		print("mismatch", np.linalg.norm(L.dot(xhat1 - xhat2)))
		assert np.linalg.norm(L.dot(xhat1 - xhat2)) < 2e-3


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


if __name__ == '__main__':
	#test_cq_center()
	test_minimax()
