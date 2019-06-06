from __future__ import print_function
import numpy as np
import psdr, psdr.demos
from checkder import check_jacobian

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
	naca = psdr.demos.NACA0012(n_upper = 2, n_lower = 2, verbose = False, maxiter = 1000 )	
	x = 0.2*np.ones(len(naca.domain))	

	if False:	
		x1 = np.copy(x)
		fx1 = naca(x1, workdir='naca_f1', keep_data = True)
		print('x1', x1)
		print('fx1', fx1)
		x2 = np.copy(x)
		x2[-1] = 0.4
		fx2 = naca(x2, workdir = 'naca_f2', keep_data = True)
		print('x2', x2)
		print('fx2', fx2)
	
	if True:
		err = check_jacobian(x, naca.eval, naca.grad, hvec = [ 1e-3])
		print('maximum error', err)
		assert err < 1e-3
if __name__ == '__main__':
	test_naca_grad()
