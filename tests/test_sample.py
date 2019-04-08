from __future__ import print_function
import numpy as np
from scipy.spatial.distance import cdist
from psdr import BoxDomain, seq_maximin_sample, fill_distance_estimate 


def test_seq_maximin_sample(m = 5):
	dom = BoxDomain(-np.ones(m), np.ones(m))
	Xhat = dom.sample(10)
	x = seq_maximin_sample(dom, Xhat, Nsamp = 100)
	assert dom.isinside(x)

def test_fill_distance(m = 5):
	dom = BoxDomain(-np.ones(m), np.ones(m))
	Xhat = dom.sample(10)
	X0 = dom.sample(1e2)
	x = seq_maximin_sample(dom, Xhat, X0 = X0)
	

	d = np.min(cdist(x.reshape(1,-1), Xhat))
	
	d2 = fill_distance_estimate(dom, Xhat, X0 = X0)
	assert np.isclose(d, d2)
