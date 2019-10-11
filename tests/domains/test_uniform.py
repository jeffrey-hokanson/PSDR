from __future__ import print_function

import numpy as np
import psdr


def test_pdf():
	m = 4
	dom = psdr.UniformDomain(-np.ones(m), np.ones(m))

	X = dom.sample(1000)
	p = dom.pdf(X)
	assert np.all(p[0] == p[1:])
