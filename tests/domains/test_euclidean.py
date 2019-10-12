from __future__ import print_function
import numpy as np
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
	assert dom._empty == False
	assert dom._point == False
	assert dom._unbounded == True
	
	# Emtpy
	dom = psdr.LinQuadDomain(lb = 1*np.ones(m), ub = -1*np.ones(m))
	assert dom.is_empty == True
	assert dom._empty == True
	assert dom._point == False
	assert dom._unbounded == False

def test_point(m = 3):
	# Unbounded
	dom = psdr.LinQuadDomain(lb = -np.inf*np.ones(m), ub = np.inf*np.ones(m))
	assert dom.is_point == False
	assert dom._empty == False
	assert dom._point == False
	assert dom._unbounded == True
	
	# Emtpy
	dom = psdr.LinQuadDomain(lb = 1*np.ones(m), ub = -1*np.ones(m))
	assert dom.is_point == False
	assert dom._empty == True
	assert dom._point == False
	assert dom._unbounded == False
	
	# Point
	dom = psdr.LinQuadDomain(lb = 1*np.ones(m), ub = 1*np.ones(m))
	assert dom.is_point == True
	assert dom._empty == False
	assert dom._point == True
	assert dom._unbounded == False

def test_unbounded(m = 3):
	# Unbounded
	dom = psdr.LinQuadDomain(lb = -np.inf*np.ones(m), ub = np.inf*np.ones(m))
	assert dom.is_unbounded == True
	assert dom._empty == False
	assert dom._point == False
	assert dom._unbounded == True

	# Empty	
	dom = psdr.LinQuadDomain(lb = 1*np.ones(m), ub = -1*np.ones(m))
	assert dom.is_unbounded == False
	assert dom._empty == True
	assert dom._point == False
	assert dom._unbounded == False
	
	# Point
	dom = psdr.LinQuadDomain(lb = 1*np.ones(m), ub = 1*np.ones(m))
	assert dom.is_unbounded == False
	assert dom._empty == False
	assert dom._point == True
	assert dom._unbounded == False
	
	# Box
	dom = psdr.LinQuadDomain(lb = -1*np.ones(m), ub = 1*np.ones(m))
	assert dom.is_unbounded == False
	assert dom._empty == False
	assert dom._point == False
	assert dom._unbounded == False


if __name__ == '__main__':
	test_is_unbounded()
