import numpy as np
import psdr



def test_sample():
	np.random.seed(0)
	m = 4
	N = 100
	X = np.random.randn(N, m)
	
	domain = psdr.ConvexHullDomain(X)
	Y = domain.sample(5)
	print(Y)
	assert Y.shape[0] == 5
	assert Y.shape[1] == m
	
	A_eq = np.random.randn(1,m)
	domain2 = domain.add_constraints( A_eq =A_eq, b_eq = [0])
	Y= domain2.sample(5)
	print(Y)
	assert Y.shape[0] == 5
	assert Y.shape[1] == m


def test_str(m = 5):
	X = np.random.randn(10, m)
	
	dom = psdr.ConvexHullDomain(X)

	# Check points are stored accurately
	assert np.all(np.isclose(dom.X, X))

	assert "ConvexHullDomain" in dom.__str__()

	# Now check with additional constraints

	A_eq = np.ones((1,m))
	b_eq = np.ones(1)
	dom = psdr.ConvexHullDomain(X, A_eq = A_eq, b_eq = b_eq)
	assert " equality " in dom.__str__()	

	A = np.ones((1,m))
	b = np.ones(1)
	dom = psdr.ConvexHullDomain(X, A = A, b = b)
	assert " inequality " in dom.__str__()

	Ls = [np.eye(m),]
	ys = [np.zeros(m),]
	rhos = [1.,]

	dom = psdr.ConvexHullDomain(X, Ls = Ls, ys = ys, rhos = rhos)
	assert " quadratic " in dom.__str__()

def test_to_linineq_1d():
	X = np.array([-1,1]).reshape(2,1)
	dom = psdr.ConvexHullDomain(X)
	dom2 = dom.to_linineq()

	assert np.all(np.isclose(dom2.lb, -1))
	assert np.all(np.isclose(dom2.ub, 1))

def test_tensor():
	X = [[1,1], [1,-1], [-1,1], [-1,-1]]
	dom1 = psdr.ConvexHullDomain(X)
	dom2 = psdr.BoxDomain(-1,1)

	dom = dom1 * dom2
	p = np.ones(3)
	x = dom.corner(p)
	print(x)
	assert np.all(np.isclose(x, [1,1,1]))


if __name__ == '__main__':
	test_sample()
