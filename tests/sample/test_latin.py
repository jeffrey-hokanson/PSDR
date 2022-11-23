import numpy as np
import psdr
import pytest

def test_latin_hypercube_maximin(m = 3, N = 5):
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	
	X = psdr.latin_hypercube_maximin(dom, N, maxiter = 100)

	assert all(dom.isinside(X))

def test_latin_hypercube_box():
	N = 1
	domain = psdr.BoxDomain(-np.ones(3), np.ones(3))
	X = psdr.latin_hypercube_sample(domain, N)
	print(X)

if __name__ == '__main__':
	test_latin_hypercube_box()

