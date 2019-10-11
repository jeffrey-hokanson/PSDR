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
	
