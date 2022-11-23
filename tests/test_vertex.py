from __future__ import print_function
import numpy as np
from scipy.spatial.distance import cdist 
from psdr import voronoi_vertex_sample, voronoi_vertex, BoxDomain
from psdr import cdist

def check_voronoi(dom, Xhat, V, L = None, randomize = True):
	if L is None:
		L = np.eye(len(dom))
	Lrank = np.linalg.matrix_rank(L)
	
	print("V shape", V.shape)
	print("All the points returned should be inside of the domain")
	I = ~dom.isinside(V)
	print(V[I])
	assert np.all(dom.isinside(V)), "All voronoi vertices must be inside the domain" 

	if randomize:
		m_constraints = len(dom)
	else:
		m_constraints = Lrank
	
	print("All points should satisfy m=%d constraints" % m_constraints)
	A = dom.A_aug
	b = dom.b_aug
	D = cdist(V, Xhat, L)
	for i, v in enumerate(V):
		# Number of hyper planne constraints
		n_hyper = np.sum(np.isclose(np.min(D[i,:]), D[i,:]))-1
		n_eq = dom.A_eq.shape[0]
		# Active inequality constraints
		n_ineq = np.sum(np.isclose(A.dot(v), b))
		print('%4d : total %4d | hyper plane %3d | equality %3d | inequality %3d' % 
			(i, n_hyper+n_eq+n_ineq, n_hyper, n_eq, n_ineq))
		print("vertex", v)
		print("distances", np.sort(D[i,:] - np.min(D[i,:])))
		assert n_hyper + n_eq + n_ineq >= m_constraints


def check_vertex_sample(dom, Xhat, X0, L = None, randomize = True):
	V = voronoi_vertex_sample(dom, Xhat, X0, L = L, randomize = randomize )
	check_voronoi(dom, Xhat, V, L = L, randomize = randomize)

def check_vertex(dom, Xhat, L = None):
	V = voronoi_vertex(dom, Xhat, L = L )
	check_voronoi(dom, Xhat, V, L = L)

def test_vertex_box(m = 5):
	np.random.seed(0)
	dom = BoxDomain(-np.ones(m), np.ones(m))
	Xhat = dom.sample(10)
	X0 = dom.sample(100)
	check_vertex_sample(dom, Xhat, X0)

def test_vertex_weight(m = 5):
	np.random.seed(0)
	dom = BoxDomain(-np.ones(m), np.ones(m))
	Xhat = dom.sample(10)
	X0 = dom.sample(100)
	L = np.diag(1./np.arange(1,m+1))
	check_vertex_sample(dom, Xhat, X0, L = L)

def test_vertex_low_rank(m = 5):
	np.random.seed(0)
	dom = BoxDomain(-np.ones(m), np.ones(m))
	Xhat = dom.sample(10)
	X0 = dom.sample(100)
	L = np.diag(1./np.arange(1,m+1))
	L[0,0] = 0.
	check_vertex_sample(dom, Xhat, X0, L = L)

def test_vertex_low_rank_nonrandomize(m = 5):
	np.random.seed(0)
	dom = BoxDomain(-np.ones(m), np.ones(m))
	Xhat = dom.sample(10)
	X0 = dom.sample(100)
	L = np.diag(1./np.arange(1,m+1))
	L[0,0] = 0.
	check_vertex_sample(dom, Xhat, X0, L = L, randomize = False)

def test_vertex_rectangular(m = 5):
	np.random.seed(0)
	dom = BoxDomain(-np.ones(m), np.ones(m))
	Xhat = dom.sample(10)
	X0 = dom.sample(100)
	L = np.ones((1, m))
	check_vertex_sample(dom, Xhat, X0, L = L)

def test_vertex_eq(m = 5):
	np.random.seed(0)
	dom = BoxDomain(-np.ones(m), np.ones(m))
	dom = dom.add_constraints(A_eq = np.ones(m), b_eq = [0])
	Xhat = dom.sample(10)
	X0 = dom.sample(5)
	check_vertex_sample(dom, Xhat, X0)


def test_vertex_full(m = 2):
	np.random.seed(0)
	dom = BoxDomain(-np.ones(m), np.ones(m))
	
	Xhat = dom.sample(10)
	#check_vertex(dom, Xhat)

	# Check with degenerate points
	Xhat[1] = Xhat[0]
	#check_vertex(dom, Xhat)

	# Check with a Lipschitz matrix
	np.random.seed(0)
	Xhat = dom.sample(5)
	#L = np.random.randn(m,m)
	L = np.diag(np.arange(1, m+1))
	print(L)
	print("Checking with a Lipschitz matrix")
	check_vertex(dom, Xhat, L = L)
		

if __name__ == '__main__':
	test_vertex_full()
	#test_vertex_rectangular()
	#test_vertex_weight()
