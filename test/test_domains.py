import numpy as np
from psdr import BoxDomain



def test_box(m=10):
	dom = BoxDomain(-np.ones(m), np.ones(m))
	assert len(dom) == m
	
	x = dom.corner(np.ones(m))
	assert np.all(np.isclose(x, np.ones(m)))


	A = np.ones((1,m))
	b = np.zeros(1)
	dom2 = dom.add_constraint(A_eq = A, b_eq = b)
	x = dom2.sample()
	assert np.isclose(np.dot(A, x), b) 
