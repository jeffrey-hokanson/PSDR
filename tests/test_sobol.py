from __future__ import print_function
import numpy as np
import psdr

def test_sobol(m=3):
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	# Triangular domain
	dom = dom.add_constraints(A = np.ones((1,m)), b = [0])
	
	X = dom.sobol_sequence(100)
	assert len(X) == 100
	assert np.all(dom.isinside(X))

if __name__ == '__main__':
	test_sobol()
