from __future__ import print_function

import numpy as np
import psdr, psdr.demos


def test_U():
	np.random.seed(0)
	fun = psdr.demos.OTLCircuit()
	X = fun.domain.sample(100)
	grads = fun.grad(X)
	lip = psdr.DiagonalLipschitzMatrix()
	lip.fit(grads = grads)
	print(lip.U)
	assert lip.U[0,0] == 1
	assert lip.U[1,1] == 1
	assert lip.U[2,2] == 1
	assert lip.U[3,3] == 1
	assert lip.U[5,4] == 1
	assert lip.U[4,5] == 1

if __name__ == '__main__':
	test_U()
