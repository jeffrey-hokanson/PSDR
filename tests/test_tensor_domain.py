import numpy as np
from psdr import TensorProductDomain, BoxDomain, LinQuadDomain

def test_sample():
	dom = TensorProductDomain([BoxDomain([-1],[1]), BoxDomain([-2],[2])])
	X = dom.sample(10)
	assert np.all(np.abs(X[:,0]) <= 1 )
	assert np.all(np.abs(X[:,1]) <= 2 )

def test_nested():
	dom1 = TensorProductDomain([BoxDomain([-1],[1]), BoxDomain([-2],[2])])
	dom2 = TensorProductDomain([BoxDomain([-1],[1]), BoxDomain([-2],[2])])
	dom = dom1 * dom2
	assert len(dom.domains) == 4
	assert len(dom) == 4

def test_closest_point(m = 5):
	Ls = [np.eye(m)]
	ys = [np.zeros(m)]
	rhos = [1]
	dom1 = LinQuadDomain(Ls = Ls, ys = ys, rhos = rhos)
	
	dom2 = BoxDomain([2],[3])
	dom = dom1 * dom2	
	x0 = np.zeros(m+1)
	x = dom.closest_point(x0)
	assert np.all(np.isclose(x[0:m],0))
	assert np.all(np.isclose(x[-1], 2))
