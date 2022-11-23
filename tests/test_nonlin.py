import numpy as np
import psdr, psdr.demos

def test_measure_nonlinearity():
	np.random.seed(0)
	fun = psdr.demos.Borehole()
	X = fun.domain.sample(100)
	fX = fun(X)

	nonlin, a = psdr.measure_nonlinearity(X, fX, verbose = True)
	print(nonlin)
	print(a)
	assert nonlin >= 0 and nonlin <= 1., "output should be in the range [0,1]"

if __name__ == '__main__':
	test_measure_nonlinearity()

