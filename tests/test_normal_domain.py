from __future__ import division
import numpy as np
from psdr import NormalDomain

def test_sampling(m = 5):
	mean = np.random.randn(m)
	L = np.random.randn(m,m)
	cov = L.dot(L.T)
	
	dom = NormalDomain(mean, cov)

	X = dom.sample(1e6)

	mean_est = np.mean(X, axis = 0)
	print('mean')
	print mean
	print mean_est
	print np.isclose(mean, np.mean(X, axis = 0), rtol = 50/np.sqrt(X.shape[0]), atol = 50/np.sqrt(X.shape[0]))
	assert np.all(np.isclose(mean, np.mean(X, axis = 0), rtol = 50/np.sqrt(X.shape[0]), atol = 50/np.sqrt(X.shape[0]))), "Mean not correct"

	print "Covariance"
	cov_est = np.cov(X.T)
	print cov	
	print cov_est
	print np.isclose(cov, cov_est, rtol = 100/np.sqrt(X.shape[0]), atol = 100/np.sqrt(X.shape[0]))
	assert np.all(np.isclose(cov, cov_est, rtol = 100/np.sqrt(X.shape[0]), atol = 100/np.sqrt(X.shape[0]))), "Covariance not correct"

	# Now check with a truncation parameter
	truncate = 1e-2
	dom2 = NormalDomain(mean, cov, truncate = truncate)
	print dom2.clip
	X2 = dom2.sample(1e5)	
	assert np.all(dom2.isinside(X2)), "Sampling did not place all points inside the constraints"
	
	frac_inside = np.sum(dom2.isinside(X))/len(X)

	assert np.isclose(1-frac_inside, truncate, atol = 100*truncate/np.sqrt(X.shape[0])), "Constraint doesn't seem to get right fraction included"

def test_normalized_domain(m = 5):
	mean = np.random.randn(m)
	L = np.random.randn(m,m)
	cov = L.dot(L.T)
	
	dom = NormalDomain(mean, cov, truncate = 1e-2)
	dom_norm = dom.normalized_domain()
	print dom_norm.norm_lb
	assert np.all(np.isclose(dom_norm.norm_lb, -np.ones(len(dom_norm))))
	print dom_norm.norm_ub
	assert np.all(np.isclose(dom_norm.norm_ub, np.ones(len(dom_norm))))
	print dom_norm.mean
	assert np.all(np.isclose(dom_norm.mean, np.zeros(len(dom_norm))))

	X_norm = dom_norm.sample(1e4)
	X = dom.unnormalize(X_norm)
	assert np.all(dom.isinside(X))

def test_corner():
	dom = NormalDomain([5], [[0.1]], truncate = 1e-2)
	print dom.norm_lb
	print dom.norm_ub
	print dom._center()
	print dom._use_norm
	dom._use_norm = True
	print "lower bound", dom.corner([-1])
	print "upper bound", dom.corner([1])
	print dom.sample(10)
	#X = dom.sample(10)
	#assert np.all(X<= dom.ub)
	#assert np.all(dom.lb <= X)


# TODO: Need to implement tensor-product quadrature rules to test 
# this density function as the Monte-Carlo formula is tautological
# Int(f) = mean( f(x)/p(x))

#def test_pdf(m = 2):
#	mean = np.random.randn(m)
#	L = np.random.randn(m,m)
#	cov = L.dot(L.T)
#	mean = np.zeros(m)
#	#cov = 1*np.eye(m)	
#	#dom = NormalDomain(mean, cov)
#	X = dom.sample(1e7)
#	#x = np.linspace(-5,5, int(1e3)).reshape(-1,1)
#	#X,Y = np.meshgrid(x,x)
#	#X = np.hstack([X.flatten(), Y.flatten()])
#	p = dom.pdf(X)
#	print dom.L
#	print dom.Linv
#	print len(p)
#	p_int = np.sum(p)/(1.*X.shape[0])
#	print p_int
#	assert np.isclose(p_int, 1)
#	assert False	

	
