import numpy as np
from psdr.demos import OpenAeroStruct
import os

def test_oas():
	oas = OpenAeroStruct()
	# Load true values
	# we do this dirname trick to make sure we can find the test data
	# https://stackoverflow.com/questions/53719350/travisci-with-pytest-and-numpy-load-file-not-found
	X = np.loadtxt(os.path.join(os.path.dirname(__file__), 'oas.input'))
	Y = np.loadtxt(os.path.join(os.path.dirname(__file__), 'oas.output'))

	Ynew = oas(X)
	assert np.all(np.isclose(Y, Ynew)), "Outputs did not match"


if __name__ == '__main__':
	np.random.seed(0)
	
	# Testing data
	oas = OpenAeroStruct()
	X = oas.sample(10)
	np.savetxt('oas.input', X)
	Y = oas(X)
	np.savetxt('oas.output', Y)
