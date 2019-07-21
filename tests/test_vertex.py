from __future__ import print_function
import numpy as np
from scipy.spatial.distance import cdist 
from psdr import voronoi_vertex, BoxDomain


def check_vertex(dom, Xhat, X0, L = None, randomize = True):
	X = voronoi_vertex(dom, Xhat, X0, L = L, randomize = randomize )
	if L is None:
		L = np.eye(len(dom))
	Lrank = np.linalg.matrix_rank(L)
	
	print("All the points returned should be inside of the domain")
	I = ~dom.isinside(X)
	print(X[I])
	assert np.all(dom.isinside(X)), "All voronoi vertices must be inside the domain" 

	if randomize:
		m_constraints = len(dom)
	else:
		m_constraints = Lrank
	
	print("All points should satisfy m=%d constraints" % m_constraints)
	A = dom.A_aug
	b = dom.b_aug
	D = cdist(L.dot(X.T).T, L.dot(Xhat.T).T)
	for i, x in enumerate(X):
		# Number of hyper planne constraints
		n_hyper = np.sum(np.isclose(np.min(D[i,:]), D[i,:]))-1
		n_eq = dom.A_eq.shape[0]
		# Active inequality constraints
		n_ineq = np.sum(np.isclose(A.dot(x), b))
		print('%4d : total %4d | hyper plane %3d | equality %3d | inequality %3d' % 
			(i, n_hyper+n_eq+n_ineq, n_hyper, n_eq, n_ineq))
		print(x)
		print(np.sort(D[i,:] - np.min(D[i,:])))
		assert n_hyper + n_eq + n_ineq >= m_constraints

def test_vertex_box(m = 5):
	np.random.seed(0)
	dom = BoxDomain(-np.ones(m), np.ones(m))
	Xhat = dom.sample(10)
	X0 = dom.sample(100)
	check_vertex(dom, Xhat, X0)

def test_vertex_weight(m = 5):
	np.random.seed(0)
	dom = BoxDomain(-np.ones(m), np.ones(m))
	Xhat = dom.sample(10)
	X0 = dom.sample(100)
	L = np.diag(1./np.arange(1,m+1))
	check_vertex(dom, Xhat, X0, L = L)

def test_vertex_low_rank(m = 5):
	np.random.seed(0)
	dom = BoxDomain(-np.ones(m), np.ones(m))
	Xhat = dom.sample(10)
	X0 = dom.sample(100)
	L = np.diag(1./np.arange(1,m+1))
	L[0,0] = 0.
	check_vertex(dom, Xhat, X0, L = L)

def test_vertex_low_rank_nonrandomize(m = 5):
	np.random.seed(0)
	dom = BoxDomain(-np.ones(m), np.ones(m))
	Xhat = dom.sample(10)
	X0 = dom.sample(100)
	L = np.diag(1./np.arange(1,m+1))
	L[0,0] = 0.
	check_vertex(dom, Xhat, X0, L = L, randomize = False)

def test_vertex_rectangular(m = 5):
	np.random.seed(0)
	dom = BoxDomain(-np.ones(m), np.ones(m))
	Xhat = dom.sample(10)
	X0 = dom.sample(100)
	L = np.ones((1, m))
	check_vertex(dom, Xhat, X0, L = L)

def test_vertex_eq(m = 5):
	np.random.seed(0)
	dom = BoxDomain(-np.ones(m), np.ones(m))
	dom = dom.add_constraints(A_eq = np.ones(m), b_eq = [0])
	Xhat = dom.sample(10)
	X0 = dom.sample(5)
	check_vertex(dom, Xhat, X0)

if __name__ == '__main__':
	test_vertex_rectangular()
	#test_vertex_weight()
