from __future__ import print_function

import numpy as np

import psdr


def test_projection(m = 3, N = 5):
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))

	Ls = [np.random.randn(1,m) for i in range(m)]

	X = psdr.projection_sample(dom, N, Ls, maxiter = 10)
	print(X)
	assert all(dom.isinside(X)), "Samples not in the domain"

if __name__ == '__main__':
	test_projection()	
