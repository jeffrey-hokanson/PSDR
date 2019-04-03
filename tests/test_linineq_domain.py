from __future__ import print_function
import numpy as np
from psdr import BoxDomain, LinIneqDomain


def test_cheb(m=5):
	lb = -np.ones(m)
	ub = np.ones(m)
	dom = BoxDomain(lb, ub)
	A = np.ones((1,m))
	b = np.zeros(1,)
	
	dom2 = dom.add_constraints(A,b)

	center, radius = dom2.chebyshev_center()
	print(center)
	print(radius)
	
	assert dom2.isinside(center), "Center must be inside"
	
	for i in range(100):
		p = np.random.randn(m)
		p /= np.linalg.norm(p)
		assert dom2.extent(center, p) >= radius, "violated radius assumption"
