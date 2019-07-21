from __future__ import print_function

import numpy as np
import psdr

def test_seq_maximin(m = 3):
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	Xhat = dom.sample(10)
	x = psdr.seq_maximin_sample(dom, Xhat, Nsamp = 100)
	assert dom.isinside(x)

