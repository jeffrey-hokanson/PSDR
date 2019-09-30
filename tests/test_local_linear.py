from __future__ import print_function, division
import numpy as np
import psdr, psdr.demos
from scipy.spatial.distance import cdist

def test_perplexity_bandwidth():
	borehole = psdr.demos.Borehole()
	X = borehole.domain.sample(100)
	x = borehole.domain.sample()
	d = cdist(X, x.reshape(1,-1), 'sqeuclidean').flatten()
	perplexity = 10
	beta = psdr.perplexity_bandwidth(d, perplexity)
	# Check perplexity
	p = np.exp(-beta*d)
	p /= np.sum(p)
	perplexity_computed = np.exp(-np.sum(p*np.log(p)))
	print("target", perplexity, "found", perplexity_computed)
	assert np.isclose(perplexity, perplexity_computed)

def test_local_linear_grads():
	m = 6
	a = np.random.randn(m)
	fun = lambda x: a.dot(x)
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	func = psdr.Function(fun, dom)

	# This should accurately estimate a linear model 
	X = func.domain.sample(20)
	fX = func(X)
	grads = psdr.local_linear_grads(X, fX)
	for grad in grads:
		assert np.all(np.isclose(grad, a))


if __name__ == '__main__':
	#test_perplexity_bandwidth()
	test_local_linear_grad()
