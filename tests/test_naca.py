from __future__ import print_function
import numpy as np
import psdr, psdr.demos
from checkder import check_jacobian
import os
import multiprocessing

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

def test_naca_grad():
	nprocesses = max(1, multiprocessing.cpu_count() - 1)

	try:
		travis = os.environ['TRAVIS'] == 'true'
	except KeyError:
		travis = False

	if travis:
		# We limit number of dimensions and iteration count 
		# so entire test takes less than 10min on Travis-CI
		naca = psdr.demos.NACA0012(n_upper = 1, n_lower = 1, verbose = True, maxiter = 200, nprocesses = 1)	
		x = 0.2*np.ones(len(naca.domain))	
		err = check_jacobian(x, naca.eval, naca.grad, hvec = [ 1e-4])
		print('maximum error', err)
		assert err < 5e-2
	else:
		naca = psdr.demos.NACA0012(n_upper = 1, n_lower = 1, verbose = True, maxiter = 1000, nprocesses = nprocesses)	
		x = 0.2*np.ones(len(naca.domain))	
		err = check_jacobian(x, naca.eval, naca.grad, hvec = [ 1e-4])
		print('maximum error', err)
		assert err < 5e-4
		
if __name__ == '__main__':
	test_naca_grad()
