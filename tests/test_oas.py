import numpy as np
from psdr.demos import OpenAeroStruct


def test_oas():
	oas = OpenAeroStruct()
	X = np.loadtxt('oas.input')
	Y = np.loadtxt('oas.output')

	Ynew = oas(X)
	assert np.isclose(Y, Ynew), "Outputs did not match"


if __name__ == '__main__':
	np.random.seed(0)
	
	# Testing data
	oas = OpenAeroStruct()
	X = oas.sample(10)
	np.savetxt('oas.input', X)
	Y = oas(X)
	np.savetxt('oas.output', Y)
