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


if __name__ == '__main__':
	test_is_unbounded()
