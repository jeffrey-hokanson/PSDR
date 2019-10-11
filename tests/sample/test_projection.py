from __future__ import print_function

import numpy as np

import psdr


def test_projection_full(m = 3, N = 5):
	np.random.seed(0)
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))

	Ls = [np.random.randn(1,m) for i in range(m)]

	X = psdr.projection_sample(dom, N, Ls, maxiter = 10)
	print(X)
	assert all(dom.isinside(X)), "Samples not in the domain"

def test_projection_low(m = 3, N = 5):
	np.random.seed(0)
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))

	Ls = [np.random.randn(1,m) for i in range(m-1)]

	X = psdr.projection_sample(dom, N, Ls, maxiter = 10)
	assert all(dom.isinside(X)), "Samples not in the domain"

def test_projection_lhs(m = 3, N = 5):
	np.random.seed(0)
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	X = psdr.projection_sample(dom, N, np.eye(m), _lhs = True)
	assert all(dom.isinside(X)), "Samples not in the domain"

def test_projection_large(m = 2, N = 3):
	np.random.seed(0)
	# Try to run through the whole sampler so the SAT has no more solutions to try
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	X = psdr.projection_sample(dom, N, np.eye(m), _lhs = True, maxiter = 10000)
	assert all(dom.isinside(X)), "Samples not in the domain"

if __name__ == '__main__':
	test_projection_low()	
