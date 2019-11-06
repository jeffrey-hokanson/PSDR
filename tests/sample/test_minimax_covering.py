from __future__ import print_function
import numpy as np
from scipy.spatial.distance import cdist

import psdr

def test_minimax_covering(m = 2):
	np.random.seed(0)

	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	r = 0.5

	X = psdr.minimax_covering(dom, r)
	X2 = dom.sample(1e5)
	D = cdist(X, X2)
	min_dist = np.max(np.min(cdist(X, X2), axis = 0))
	print("minimum distance %5.2e; target %5.2e" % (min_dist, r))
	assert min_dist < r, "Sampling did not meet target separation"

if __name__ == '__main__':
	test_minimax_covering()	
