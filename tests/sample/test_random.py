import numpy as np
import psdr

def test_random():
	m = 5
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	
	X = psdr.random_sample(dom, 10)
	assert np.all(dom.isinside(X))
