import numpy as np
from psdr import LogNormalDomain

def test_sample():
	mean = 0
	cov = 10
	scaling = 1
	offset = 0.
	truncate = 1e-2

	dom = LogNormalDomain(mean, cov, offset, scaling, truncate)

	X = dom.sample(100)
	print dom.isinside(X)
	assert np.all(dom.isinside(X))

def test_normalize():
	mean = 0
	cov = 10
	scaling = 1
	offset = 0.
	truncate = 1e-2

	dom = LogNormalDomain(mean, cov, offset, scaling, truncate)
	dom_norm = dom.normalized_domain()
	print dom_norm.lb
	print dom_norm.ub
	assert np.isclose(dom_norm.lb, -1)
	assert np.isclose(dom_norm.ub, 1)

	# TODO: Check densities
	X = dom.sample(100)
	X_norm = dom.normalize(X)
	p = dom.pdf(X)
	p_norm = dom_norm.pdf(X_norm)
	assert np.all(np.isclose(p, p_norm))
