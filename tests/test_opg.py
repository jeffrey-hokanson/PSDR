from __future__ import print_function
import numpy as np
import scipy.linalg
import psdr
from psdr.demos import QuadraticFunction

def test_opg():
	np.random.seed(0)
	A = np.random.randn(5,2)
	fun = QuadraticFunction(quad = A.dot(A.T))
	X = fun.domain.sample(500)
	fX = fun(X)

	for standardize in [True, False]: 
		opg = psdr.OuterProductGradient(standardize = standardize)
		opg.fit(X, fX)

		ang = scipy.linalg.subspace_angles(A, opg.U[:,0:2])
		print("subspace angles", np.rad2deg(ang))
		assert ang[0] < 0.1

if __name__ == '__main__':
	test_opg()		
