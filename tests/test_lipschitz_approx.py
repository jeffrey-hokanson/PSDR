import numpy as np
import psdr
import psdr.demos
from psdr.lipschitz_approx import *
from scipy.linalg import subspace_angles
import pytest


def check_lipschitz_eval(L, X, fX):
	I, J = np.tril_indices(len(X), k=1)
	rhs = np.sqrt(np.sum( (L @ (X[I] - X[J]).T)**2, axis = 0))
	lhs = np.abs(fX[I] - fX[J])
	return np.max(lhs - rhs, 0)


@pytest.mark.parametrize("norm", [1,2,np.inf])
@pytest.mark.parametrize("epsilon", [0, 0.1])
def test_lipschitz_approx(norm, epsilon):
	fun = psdr.demos.Borehole()
	X = fun.domain.sample_grid(2)
	lip = psdr.LipschitzMatrix()
	lip.fit(grads = fun.grad(X))

	# Now generate some random data
	X = fun.domain.sample(50)
	fX = fun(X)

	# Implement for multiple ranks 
	for i in range(len(fun.domain)):
		U = lip.U[:,:-i]
		LUU = lip.L @ U @ U.T

		np.random.seed(i)
		fX += epsilon* np.random.randn(*fX.shape)	
		y = lipschitz_approximation_compatible_data(LUU, X, fX[:,0], norm = norm, verbose = True)

		err = check_lipschitz_eval(LUU, X, y)
		print("error in data", err)
		assert err < 1e-5


def test_lipschitz_approx_class():
	r""" Integration testing
	"""
	fun = psdr.demos.OTLCircuit()
	X = fun.domain.sample_grid(2)
	lip = psdr.LipschitzMatrix()
	lip.fit(grads = fun.grad(X))
	
		
	lipapprox = LipschitzApproximation(lip.L, fun.domain)
	X = fun.domain.sample(40)
	fX = fun(X)
	#fX += 0.1*np.random.randn(*fX.shape)
	lipapprox.fit(X, fX[:,0])

	y = lipapprox(X)
	print(fX[:,0] - y)
	err = np.max(np.abs(fX[:,0] - y))
	assert err< 1e-9
	
	# Check identification of active subspace
	for i in range(1,len(fun.domain)):
		U = lip.U[:,:-i]
		LUU = lip.L @ U @ U.T
		lipapprox = LipschitzApproximation(LUU, fun.domain)
		print(U.shape)
		print(lipapprox.U.shape)
		ang = subspace_angles(U, lipapprox.U)
		print(ang)
		assert np.max(ang) < 1e-7, "Did not correctly identify active subspace"	

if __name__ == '__main__':
	#test_lipschitz_approx(1, 0)
	test_lipschitz_approx_class()

