import numpy as np
import psdr
from psdr import BoxDomain, ConvexHullDomain, LinIneqDomain

def test_isinside(m = 5):
	np.random.seed(0)
	dom = BoxDomain(-10*np.ones(m), -5*np.ones(m))
	
	X = dom.sample(10)

	Xg = dom.sample_grid(2)
	hull = ConvexHullDomain(Xg)
	
	assert np.all(hull.isinside(X))

def test_sphere(m = 3):
	np.random.seed(0)
	X = psdr.geometry.sample_sphere(m, 50)
	hull = ConvexHullDomain(X)
	dom = hull.to_linineq()  

	# Test extent
	for i in range(5):
		x = dom.sample()
		p = np.random.randn(m)

		alpha1 = hull.extent(x, p)
		alpha2 = dom.extent(x,p)
		print(alpha1,alpha2)
		assert np.isclose(alpha1, alpha2)

	# Test corner
	for i in range(5):
		p = np.random.randn(m)
		c1 = hull.corner(p)
		c2 = dom.corner(p)
		print('c1', c1)
		print('c2', c2)
		assert np.all(np.isclose(c1, c2))	

	# Test closest point
	for i in range(5):
		x = 2*psdr.geometry.sample_sphere(m, 1)
		L = np.random.randn(m,m)
		c1 = hull.closest_point(p, L = L)
		c2 = dom.closest_point(p, L = L)
		print('c1', c1)
		print('c2', c2)
		assert np.linalg.norm(c1 - c2) < 1e-5	


def test_constraints(m=3):
	np.random.seed(0)

	dom = BoxDomain(-1*np.ones(m), np.ones(m))
	
	# Lower pyramid portion
	dom_con = dom.add_constraints(A = np.ones((1,m)), b = np.ones(1))
	
	# Convex hull describes the same space as dom_con
	X = dom.sample_grid(2)
	hull = ConvexHullDomain(X, A = dom_con.A, b = dom_con.b)

	# Check that the same points are inside
	X = dom.sample(100)
	assert np.all(hull.isinside(X) == dom_con.isinside(X))

	# Check sampling
	X = hull.sample(100)
	assert np.all(dom_con.isinside(X))


if __name__ == '__main__':
	test_sphere()
