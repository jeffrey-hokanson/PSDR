import numpy as np

import psdr


def test_latin_hypercube_maximin(m = 3, N = 5):
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	
	X = psdr.latin_hypercube_maximin(dom, N, maxiter = 100)

	assert all(dom.isinside(X))

if __name__ == '__main__':
	test_latin_hypercube_maximin_box()

