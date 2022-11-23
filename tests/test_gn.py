from __future__ import print_function
import numpy as np
from psdr.gn import gauss_newton

def test_gn():
	# NLS example taken from Wikipedia:
	#   https://en.wikipedia.org/wiki/Gauss-Newton_algorithm#Example
	# Substrate concentration
	s = np.array([0.038, 0.194, 0.425, 0.626, 1.253, 2.500, 3.740])
	# Reaction rate
	r = np.array([0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317])
	# Rate model
	m = lambda p, s: p[0] * s / (p[1] + s)
	f = lambda p: m(p, s) - r
	F = lambda p: np.c_[s / (p[1] + s), -p[0] * s / (p[1] + s) ** 2]

	x0 = np.zeros((2,))
	x = gauss_newton(f, F, x0, verbose=1)

if __name__ == '__main__':
	test_gn()
