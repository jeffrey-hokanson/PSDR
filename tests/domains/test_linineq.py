import numpy as np
import psdr

def test_chebyshev(m = 5):
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	assert np.isclose(dom.radius, 1.)
	
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	assert np.all(np.isclose(dom.center, np.zeros(m)))	
