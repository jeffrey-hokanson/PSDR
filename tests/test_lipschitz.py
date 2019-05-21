from __future__ import print_function
import numpy as np
import scipy.linalg
from psdr import LipschitzMatrix, LipschitzConstant, DiagonalLipschitzMatrix, BoxDomain
from psdr.demos import OTLCircuit

from checkder import *

np.random.seed(0)

def test_lipschitz_grad(N = 10):

	func = OTLCircuit()
	X = func.domain.sample(N)
	grads = func.grad(X)
	

	lip_mat = LipschitzMatrix(ftol = 1e-10, gtol = 1e-10)
	lip_diag = DiagonalLipschitzMatrix(ftol = 1e-10, gtol = 1e-10)
	lip_const = LipschitzConstant()
	
	
	for lip in [lip_mat, lip_diag, lip_const]: 
		lip.fit(grads = grads)
		H = np.copy(lip.H)

		for g in grads:
			gap = np.min(scipy.linalg.eigvalsh(H - np.outer(g,g)))
			print(gap)
			assert gap >= -1e-6
		
def test_lipschitz_func(M = 20):

	func = OTLCircuit()
	X = func.domain.sample(M)
	fX = func(X)


	lip_mat = LipschitzMatrix(ftol = 1e-10, gtol = 1e-10)
	lip_diag = DiagonalLipschitzMatrix(ftol = 1e-10, gtol = 1e-10)
	lip_const = LipschitzConstant()

	for lip in [lip_mat, lip_diag, lip_const]: 
	
		lip.fit(X, fX)
		
		H = np.copy(lip.H)

		for i in range(M):
			for j in range(i+1,M):
				y = X[i] - X[j]
				gap = y.dot(H.dot(y)) - (fX[i] - fX[j])**2
				print(gap)
				assert gap >= -1e-6



def test_solver(N = 50, M = 0):
	func = OTLCircuit()
	X = func.domain.sample(N)
	grads = func.grad(X)

	X = func.domain.sample(M)
	fX = func(X)
	X = None
	fX = None	
	
	lip = LipschitzMatrix(verbose = True, method = 'cvxpy')
	lip.fit(X = X, fX = fX, grads = grads)
	H1 = np.copy(lip.H)

	lip = LipschitzMatrix(verbose = True, method = 'param')
	lip.fit(X = X, fX = fX, grads = grads)
	H2 = np.copy(lip.H)
	
	lip = LipschitzMatrix(verbose = True, method = 'cvxopt')
	lip.fit(X = X, fX = fX, grads = grads)
	H3 = np.copy(lip.H)

	
	for g in grads:
		gap1 = np.min(scipy.linalg.eigvalsh(H1 - np.outer(g,g)))
		gap2 = np.min(scipy.linalg.eigvalsh(H2 - np.outer(g,g)))
		gap3 = np.min(scipy.linalg.eigvalsh(H3 - np.outer(g,g)))
		print("%7.2e \t %7.2e \t %7.2e" % (gap1,gap2, gap3))
		assert gap1 >= -1e-6
		assert gap2 >= -1e-6
		assert gap3 >= -1e-6

	#print(H1)
	print(np.trace(H1))
	#print(H2)
	print(np.trace(H2))
	print(np.trace(H3))
	print(np.linalg.norm(H1- H2, 'fro'))
	assert np.isclose(np.linalg.norm(H1 - H2, 'fro'),0)

def test_set_bounds():
	from psdr.lipschitz import LowerBound, UpperBound
	L = np.eye(2)
	L[1,1] = 0.1
	dom = BoxDomain([-1,-1],[1,1])
	X = dom.sample(10)
	fX = np.random.randn(X.shape[0])
	
	lower = LowerBound(L, X, fX)
	x = dom.sample()
	err = check_gradient(x, lower.eval, lower.grad) 		
	assert err < 1e-7, "Gradient error too large"	
	
	upper = UpperBound(L, X, fX)
	x = dom.sample()
	err = check_gradient(x, upper.eval, upper.grad) 		
	assert err < 1e-7, "Gradient error too large"

def test_lipschitz_bounds():
	fun = OTLCircuit()
	X = fun.domain.sample_grid(2)
	fX = fun(X)
	grads = fun.grad(X)
	lip = LipschitzMatrix()
	lip.fit(grads = grads)
	L = lip.L

	Xtest = fun.domain.sample(10)
	lb, ub = lip.bounds(X, fX, Xtest)
	for i, x in enumerate(Xtest):
		assert np.isclose(lb[i], np.max([fX[j] - np.linalg.norm(L.dot(X[j] - x)) for j in range(len(X)) ]))
		assert np.isclose(ub[i], np.min([fX[j] + np.linalg.norm(L.dot(X[j] - x)) for j in range(len(X)) ]))

def test_lipschitz_bound_domain():
	fun = OTLCircuit()
	X = fun.domain.sample_grid(2)
	fX = fun(X)
	grads = fun.grad(X)
	lip = LipschitzMatrix()
	lip.fit(grads = grads)
	L = lip.L

	dom = fun.domain.add_constraints(A_eq = np.ones((1,len(fun.domain))), b_eq = [1])
	
	lb, ub = lip.bounds_domain(X, fX, dom, Nsamp = 10)

	# Compare against random samples from the domain
	Xtest = dom.sample(100)
	lbs, ubs = lip.bounds(X, fX, Xtest)
	print("lower bound", "domain", lb, "sample", np.min(lbs)) 
	print("upper bound", "domain", ub, "sample", np.max(ubs)) 
	assert lb <= np.min(lbs)
	assert ub >= np.max(ubs)

if __name__ == '__main__':
	test_lipschitz_bound_domain()
