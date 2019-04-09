from __future__ import print_function
import numpy as np
from scipy.spatial.distance import cdist, pdist
from psdr import BoxDomain, seq_maximin_sample, fill_distance_estimate, initial_sample 


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

def test_initial_sample(m = 20):
	dom = BoxDomain(-np.ones(m), np.ones(m))
	L1 = np.random.randn(1,m)
	L2 = np.random.randn(2,m)
	L3 = np.random.randn(3,m)

	Nsamp = 100
	for L in [L1, L2, L3]:
		# Standard uniform sampling
		X1 = dom.sample(Nsamp)
		LX1 = L.dot(X1.T).T
		d1 = pdist(LX1)

		# initial sample algorithm
		X2 = initial_sample(dom, L, Nsamp = Nsamp)
		assert np.all(dom.isinside(X2))
		LX2 = L.dot(X2.T).T
		d2 = pdist(LX2)
		print("uniform sampling mean distance", np.mean(d1), 'min', np.min(d1))
		print("initial sampling mean distance", np.mean(d2), 'min', np.min(d2))
		assert np.mean(d2) > np.mean(d1), "Initial sampling ineffective"	


if __name__ == '__main__':
	test_initial_sample()
