import numpy as np
from psdr import BoxDomain

def test_corner(m=10):
	dom = BoxDomain(-np.ones(m), np.ones(m))
	assert len(dom) == m
	
	x = dom.corner(np.ones(m))
	assert np.all(np.isclose(x, np.ones(m)))


