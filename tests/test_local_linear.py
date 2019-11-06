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

def test_local_linear_grads(m = 10):
	np.random.seed(0)
	a = np.random.randn(m)
	func = psdr.demos.AffineFunction(linear = a)

	# This should accurately estimate a linear model
	# regardless of the permutation of arguments 
	X = func.domain.sample(20)
	fX = func(X)

	kwargs = [{},
		{'perplexity': 10,},
		{'bandwidth': 0.1,},
		{'bandwidth': 'xia'},
	]
	for kwarg in kwargs:
		grads = psdr.local_linear_grads(X, fX, **kwarg)
		print("a", a)
		for grad in grads:
			print("g", grad)
			assert np.all(np.isclose(grad, a))


if __name__ == '__main__':
	#test_perplexity_bandwidth()
	test_local_linear_grad()
