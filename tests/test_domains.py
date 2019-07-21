import numpy as np
import psdr
from psdr import BoxDomain



def test_box(m=10):
	dom = BoxDomain(-np.ones(m), np.ones(m))
	assert len(dom) == m
	
	x = dom.corner(np.ones(m))
	assert np.all(np.isclose(x, np.ones(m)))


	A = np.ones((1,m))
	b = np.zeros(1)
	dom2 = dom.add_constraints(A_eq = A, b_eq = b)
	x = dom2.sample()
	assert np.isclose(np.dot(A, x), b) 


def test_point(m = 3):
	lb = 0*np.ones(m)
	ub = 1*np.ones(m)

	dom = BoxDomain(lb, ub)
	assert dom.is_point is False

	dom = BoxDomain(ub, ub)
	assert dom.is_point is True

	dom = BoxDomain(lb, ub)
	dom = dom.add_constraints(A_eq = np.ones((1,m)), b_eq = [ 0])
	assert dom.is_point is True

def test_sample_grid(m = 3):
	lb = 0*np.ones(m)
	ub = 1*np.ones(m)

	dom = BoxDomain(lb, ub)
	
	X = dom.sample_grid(5)
	assert len(X) == 5**m


def test_convex_hull():
	x = np.random.randn(5)
	dom = psdr.ConvexHullDomain(x)
	
	x = np.random.randn(5,1)
	dom = psdr.ConvexHullDomain(x)
