import psdr
import numpy as np


def test_maximin_coffeehouse(m = 2, N = 20):
	np.random.seed(0)
	domain = psdr.BoxDomain(-np.ones(m), np.ones(m))
	L = np.random.randn(m, m)
	Xhat = psdr.maximin_coffeehouse(domain, N, L = L, N0 = 1000)

	
	print(Xhat)
	assert np.all(domain.isinside(Xhat)), "points outside the domain"

	# We expect that points start getting closer
	dists = np.zeros(N)
	for k in range(1,N):
		dists[k] = np.min(psdr.cdist(Xhat[k], Xhat[:k], L = L))
		print(dists[k])

	assert np.all(dists[2:] - dists[1:-1]<0), "distances should be monotonically decreasing"

if __name__ == '__main__':
	test_maximin_coffeehouse()
	
