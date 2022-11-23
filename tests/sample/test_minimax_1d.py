import numpy as np
import psdr


def test_minimax_1d():
	# 1-d domain
	dom = psdr.BoxDomain(-10,10)
	X = psdr.minimax_design_1d(dom, 10)
	assert np.all(dom.isinside(X))
	# Fill distance
	dist = psdr.fill_distance_estimate(dom, X)
	print(dist)
	assert np.isclose(dist, 1.)


	# 2-d domain with 1-d Lipschitz
	dom = psdr.BoxDomain(-np.ones(4), np.ones(4))
	L = np.ones((1,4))
	X = psdr.minimax_design_1d(dom, 4, L = L)
	assert np.all(dom.isinside(X))
	print(X)
	dist = psdr.fill_distance_estimate(dom, X, L = L)
	print(dist)
	assert np.isclose(dist, 1.)
