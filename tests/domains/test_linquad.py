import numpy as np
import psdr


def test_str():
	m = 5
	lb = -np.ones(m)
	ub = np.ones(m)

	dom = psdr.BoxDomain(lb = lb, ub = ub)
	assert 'BoxDomain' in dom.__str__()

	A = np.ones((2,m))
	b = np.ones(2)
	dom = psdr.LinIneqDomain(A = A, b = b, lb = lb, ub = ub)
	assert 'LinIneqDomain' in dom.__str__()
	assert ' inequality ' in dom.__str__()

	A_eq = np.ones((1,m))
	b_eq = np.ones(1)
	dom = psdr.LinIneqDomain(A_eq = A_eq, b_eq = b_eq, lb = lb, ub = ub)
	assert 'LinIneqDomain' in dom.__str__()
	assert ' equality ' in dom.__str__()

	Ls = [np.eye(m),]
	ys = [np.zeros(m),]
	rhos = [1.,]

	dom = psdr.LinQuadDomain(Ls = Ls, ys = ys, rhos = rhos)
	assert 'LinQuadDomain' in dom.__str__()
	assert ' quadratic ' in dom.__str__()


def test_and(m = 5):
	np.random.seed(0)
	lb = -np.ones(m)
	ub = np.ones(m)
	dom1 = psdr.BoxDomain(lb, ub)

	Ls = [np.eye(m),]
	ys = [np.ones(m),]
	rhos = [0.5,]
	dom2 = psdr.LinQuadDomain(Ls = Ls, ys = ys, rhos = rhos)

	# Combine the two domains
	dom3 = dom1 & dom2

	# Check inclusion
	for it in range(10):
		p = np.random.randn(m)
		x = dom3.corner(p)
		assert dom1.isinside(x)
		assert dom2.isinside(x)

	# Now try with a tensor product domain
	lb = -np.ones(2)
	ub = np.ones(2)
	dom1a = psdr.BoxDomain(lb, ub)	
	
	lb = -np.ones(m-2)
	ub = np.ones(m-2)
	dom1b = psdr.BoxDomain(lb, ub)	
	
	dom1 = dom1a * dom1b
	
	# Combine the two domains
	dom3 = dom1 & dom2

	# Check inclusion
	for it in range(10):
		p = np.random.randn(m)
		x = dom3.corner(p)
		assert dom1.isinside(x)
		assert dom2.isinside(x)

	# Combine the two domains
	dom3 = dom2 & dom1

	# Check inclusion
	for it in range(10):
		p = np.random.randn(m)
		x = dom3.corner(p)
		assert dom1.isinside(x)
		assert dom2.isinside(x)
