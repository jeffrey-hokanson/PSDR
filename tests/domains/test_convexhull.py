import numpy as np
import psdr
import scipy.linalg
from scipy.linalg import orth
import pytest

from psdr.domains.convexhull import _hull_to_linineq


@pytest.mark.parametrize("M", [1,2,3,4,5,6,7,8,9,10])
@pytest.mark.parametrize("m", [1,2,3,4])
def test_hull_to_ineq(M, m):
	np.random.seed(0)
	X = np.random.randn(M,m)
	dom_hull = psdr.ConvexHullDomain(X)
	dom_linineq = dom_hull.to_linineq()

	for it in range(10):
		if M > 1:
			# Hit and run will error out with a point domain
			x = dom_hull._hit_and_run()
		else:
			x = dom_hull.sample()
		assert dom_linineq.isinside(x)


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


@pytest.mark.parametrize("m", [3, 5])
@pytest.mark.parametrize("n", [1,2])
def test_A_eq(m,n):
	np.random.seed(0)
	X = np.random.randn(10+m, m)
	Q = np.random.randn(m,n)
	Q = orth(Q)
	# Orthogonalize against this direction so these points are on a low-dim space
	X = X - (X @ Q) @ Q.T
	
	dom = psdr.ConvexHullDomain(X)
	Qeq = dom._A_eq_basis	
	ang = scipy.linalg.subspace_angles(Qeq, Q) 
	print(ang)
	assert np.all(np.isclose(ang, 0))



@pytest.mark.parametrize("m", [2,3,4,5])
@pytest.mark.parametrize("nullspace_dim", [0,1,2,3])
def test_A_eq_deficient(m, nullspace_dim):
	if nullspace_dim >= m: return

	X = np.random.randn(m - nullspace_dim+1, m)
	dom = psdr.ConvexHullDomain(X)
	Qeq = dom._A_eq_basis
	print(Qeq.shape)
	assert Qeq.shape[1] == nullspace_dim	



if __name__ == '__main__':
	#test_sample()
	#test_A_eq(4, 2)
	#test_A_eq_deficient(4, 0)
	test_hull_to_ineq(1, 4)
