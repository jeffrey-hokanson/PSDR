from __future__ import print_function

import numpy as np
import psdr


def test_pdf():
	m = 3
	dom = psdr.UniformDomain(-10*np.ones(m), 10*np.ones(m))

	X = dom.sample(1000)
	p = dom.pdf(X)
	assert np.all(p[0] == p[1:])

	# Check via quadrature
	X, w = dom.quadrature_rule(1000)
	p = np.sum(dom.pdf(X)*w)
	print("probability", p)
	assert np.isclose(p, 1)
