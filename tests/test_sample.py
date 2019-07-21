from __future__ import print_function
import numpy as np
from scipy.spatial.distance import cdist, pdist
import psdr
from psdr import BoxDomain, seq_maximin_sample, fill_distance_estimate, initial_sample
from psdr import SequentialMaximinSampler 
from psdr.demos import Borehole, GolinskiGearbox


def test_fill_distance(m = 5):
	dom = BoxDomain(-np.ones(m), np.ones(m))
	Xhat = dom.sample(10)
	X0 = dom.sample(1e2)
	x = seq_maximin_sample(dom, Xhat, X0 = X0)
	

	d = np.min(cdist(x.reshape(1,-1), Xhat))
	
	d2 = fill_distance_estimate(dom, Xhat, X0 = X0)
	assert np.isclose(d, d2)

def test_initial_sample(m = 10):
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

def no_test_seqmaximin():
	# As the underlying function is already tested above
	# here we are just making sure that the code doesn't error
	fun1 = Borehole()
	fun2 = GolinskiGearbox()
	for fun in [fun1, fun2]:
		L1 = np.ones( (1,len(fun.domain)) )
		for L in [None, L1]:
			print("hi")
			samp = SequentialMaximinSampler(fun, L = L)
			samp.sample(4)
			samp.sample()
			samp.sample(2)
			assert samp.X.shape == (7,len(fun.domain))
			print(samp.fX)
			assert len(samp.fX) == 7	


def test_maximin(N = 11, m = 5):
	dom = BoxDomain(-np.ones(m), np.ones(m))
	L = np.ones((1,m))
	X = psdr.maximin_sample(dom, N, L = L, maxiter = 500)
	print(X)
	d = L.dot(X.T).flatten()
	d = np.sort(d)
	print(d)
	d_expect = np.linspace(-m,m, N)
	print(d_expect)
	assert np.all(np.isclose(d, d_expect, atol = 1e-5))
		
	# Now do a 2-d version
	L = np.random.randn(2, m)
	X = psdr.maximin_sample(dom, N, L = L, maxiter = 50, verbose = True)


def no_test_lipschitz_sample(N = 5, m = 3):
	dom = BoxDomain(-np.ones(m), np.ones(m))
	# Add an inequality constraint so some combinations aren't feasible
	dom = dom.add_constraints(A = np.ones((1,m)), b = np.ones(1))
	Ls = [np.random.randn(1,m) for j in range(2)]

	# Limit the number of iterations to reduce computational cost
	X = psdr.lipschitz_sample(dom, N, Ls, maxiter = 3, jiggle = False)
	print(X)
	assert np.all(dom.isinside(X))

	# Verify that each point is distinct in projections
	for L in Ls:
		y = L.dot(X.T).T
		print(y)
		assert np.min(pdist(y)) > 0, "points not distinct in projection"
	

if __name__ == '__main__':
	#test_seq_maximin_sample()
	#test_fill_distance()
	#test_initial_sample()
	test_seqmaximin()
	#test_maximin()
	#test_lipschitz_sample()
