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

	# Check the property versions
	assert np.all(dom2.radius == radius)
	assert np.all(dom2.center == center)

def test_convex_combo(m=10):
	lb = np.zeros(m)
	ub = np.ones(m)
	A_eq = np.ones((1,m))
	b_eq = np.ones(1)

	dom = LinIneqDomain(lb = lb, ub = ub, A_eq = A_eq, b_eq = b_eq)
	c = dom.corner(np.random.randn(m))
	X = dom.sample(10)
	assert np.all(np.isclose(np.sum(X, axis = 1), 1))


if __name__ == '__main__':
	test_convex_combo()
