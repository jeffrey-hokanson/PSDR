from __future__ import print_function
import numpy as np
from scipy.linalg import subspace_angles
import psdr


def test_is_unbounded():
	m = 5
	dom = psdr.LinQuadDomain(lb = -np.inf*np.ones(m), ub = np.inf*np.ones(m))
	assert dom.is_unbounded is True

def test_corner_unbounded():
	m = 5
	dom = psdr.LinQuadDomain(lb = -np.inf*np.ones(m), ub = np.inf*np.ones(m))
	p = np.ones(m)
	try:
		dom.corner(p)
	except psdr.UnboundedDomainException:
		pass
	except:
		raise Exception("wrong error")

def test_len(m = 3):
	dom = psdr.LinQuadDomain(lb = -np.inf*np.ones(m), ub = np.inf*np.ones(m))
	assert len(dom) == m

def test_empty(m = 3):
	# Unbounded
	dom = psdr.LinQuadDomain(lb = -np.inf*np.ones(m), ub = np.inf*np.ones(m))
	assert dom.is_empty == False
	assert dom.is_point == False
	assert dom.is_unbounded == True
	
	# Emtpy
	dom = psdr.LinQuadDomain(lb = 1*np.ones(m), ub = -1*np.ones(m))
	assert dom.is_empty == True
	assert dom.is_point == False
	assert dom.is_unbounded == False

def test_point(m = 3):
	# Unbounded
	dom = psdr.LinQuadDomain(lb = -np.inf*np.ones(m), ub = np.inf*np.ones(m))
	assert dom.is_point == False
	assert dom.is_empty == False
	assert dom.is_unbounded == True
	
	# Emtpy
	dom = psdr.LinQuadDomain(lb = 1*np.ones(m), ub = -1*np.ones(m))
	assert dom.is_point == False
	assert dom.is_empty == True
	assert dom.is_unbounded == False
	
	# Point
	dom = psdr.LinQuadDomain(lb = 1*np.ones(m), ub = 1*np.ones(m))
	assert dom.is_point == True
	assert dom.is_empty == False
	assert dom.is_unbounded == False

def test_unbounded(m = 3):
	# Unbounded
	dom = psdr.LinQuadDomain(lb = -np.inf*np.ones(m), ub = np.inf*np.ones(m))
	assert dom.is_unbounded == True
	assert dom.is_empty == False
	assert dom.is_point == False

	# Empty	
	dom = psdr.LinQuadDomain(lb = 1*np.ones(m), ub = -1*np.ones(m))
	assert dom.is_unbounded == False
	assert dom.is_empty == True
	assert dom.is_point == False
	
	# Point
	dom = psdr.LinQuadDomain(lb = 1*np.ones(m), ub = 1*np.ones(m))
	assert dom.is_unbounded == False
	assert dom.is_empty == False
	assert dom.is_point == True
	
	# Box
	dom = psdr.LinQuadDomain(lb = -1*np.ones(m), ub = 1*np.ones(m))
	assert dom.is_unbounded == False
	assert dom.is_empty == False
	assert dom.is_point == False


def test_sweep(m = 5):
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))

	# Default arguments
	X, y = dom.sweep()
	assert np.all(dom.isinside(X))


	# Specify sample	
	x = dom.sample()
	X, y = dom.sweep(x = x)
	assert np.all(dom.isinside(X))
	# Check x is on the line
	dom2 = psdr.ConvexHullDomain(X)
	assert dom2.isinside(x)

	# Specify direction
	p = np.random.randn(m)
	X, y = dom.sweep(p = p)
	assert np.all(dom.isinside(X))
	d = (X[-1] - X[0]).reshape(-1,1)
	assert np.isclose(subspace_angles(d, p.reshape(-1,1)),0)

	# Check corner
	X, y = dom.sweep(p = p, corner = True)
	c = dom.corner(p)
	assert np.any([np.isclose(x, c) for x in X])

if __name__ == '__main__':
	test_is_unbounded()
