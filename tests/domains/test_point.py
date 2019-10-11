import numpy as np
import psdr


def test_point(m = 5):
	np.random.seed(0)
	x = np.random.randn(m)
	dom = psdr.PointDomain(x)

	assert len(dom) == m

	y = np.random.randn(m)
	assert np.all(np.isclose(x, dom.closest_point(y)))

	p = np.random.randn(m)
	assert np.all(np.isclose(x, dom.corner(p)))

	assert np.all(np.isclose(x, dom.lb))
	assert np.all(np.isclose(x, dom.ub))

	X = dom.sample(10)
	for xx in X:
		assert np.all(np.isclose(xx, x))


def test_and(m = 5):
	np.random.seed(0)
	dom1 = psdr.BoxDomain(-np.ones(m), np.ones(m))

	x = np.zeros(m)
	dom2 = psdr.PointDomain(x)
	
	dom3 = dom1 & dom2
	assert np.all(np.isclose(dom3.sample(), x)) 
	
	dom3 = dom2 & dom1
	assert np.all(np.isclose(dom3.sample(), x)) 

	# Check an empty domain
	x = 2*np.ones(m)
	dom2 = psdr.PointDomain(x)
	dom3 = dom1 & dom2

	try:
		y = dom3.sample()
	except psdr.EmptyDomainException:
		pass
	except:
		raise Exception("Wrong error returned")
