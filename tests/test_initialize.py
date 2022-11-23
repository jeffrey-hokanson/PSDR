from __future__ import print_function
import numpy as np
import scipy.linalg
import psdr, psdr.demos
from psdr.initialization import initialize_subspace

def test_init_subspace():
	np.random.seed(0)
	# Low rank quadratic function
	A1 = np.random.randn(5,2)
	A = A1.dot(A1.T)
	fun = psdr.demos.QuadraticFunction(quad = A)
	
	# Test only samples
	X = fun.domain.sample(500)
	fX = fun(X)
	U = initialize_subspace(X = X, fX = fX, grads = None)

	ang = scipy.linalg.subspace_angles(A1, U[:,0:2])
	print("Sample based approach")
	print("subspace angles", np.rad2deg(ang))
	assert ang[0] < 0.5, "Large subspace angle between true and estimate"

	# Test only gradients
	grads = fun.grad(X)
	U = initialize_subspace(grads = grads)
	ang = scipy.linalg.subspace_angles(A1, U[:,0:2])
	print("Gradient based approach")
	print("subspace angles", np.rad2deg(ang))
	assert ang[0] < 0.5, "Large subspace angle between true and estimate"


	# test mixed approach
	grads = fun.grad(X)
	U = initialize_subspace(X = X, fX = fX, grads = grads[0:50], n_grads = 100)
	ang = scipy.linalg.subspace_angles(A1, U[:,0:2])
	print("Mixed approach")
	print("subspace angles", np.rad2deg(ang))
	assert ang[0] < 0.5, "Large subspace angle between true and estimate"
	
if __name__ == '__main__':
	test_init_subspace()
