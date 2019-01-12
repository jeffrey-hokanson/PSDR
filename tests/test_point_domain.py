from __future__ import print_function
import numpy as np
from psdr import PointDomain

def test_point_domain_isinside():
	x = np.ones(10)
	y = np.zeros(10)
	dom = PointDomain(x)
	X = np.vstack([x,y])
	print(X.shape)
	print(dom.isinside(X))
	assert np.all(dom.isinside(X) == np.array([True,False]))
