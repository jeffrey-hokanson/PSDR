from __future__ import print_function
import numpy as np
import psdr, psdr.demos


def test_naca():
	naca = psdr.demos.NACA0012()
	x = naca.domain.sample()
	x = x*0.
	y = naca(x)
	print('output', y)
	y_true = np.array([0.3269333, 0.02134973])
	print('true  ', y_true)
	assert np.all(np.isclose(y, y_true))

	# test with verbose output on a smaller domain
	naca = psdr.demos.NACA0012(n_lower = 3, n_upper = 4)
	x = naca.domain.sample()
	x = 0.*x
	y = naca(x, verbose = True)
	print('output', y)
	y_true = np.array([0.3269333, 0.02134973])
	print('true  ', y_true)
	assert np.all(np.isclose(y, y_true))
	

if __name__ == '__main__':
	test_naca()
