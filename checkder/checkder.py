r""" A derivative checking library
"""

from __future__ import print_function

import numpy as np
from numpy.linalg import norm

def check_jacobian(x, residual, jacobian, hvec = None):
	n = x.shape[0]
	if hvec is None:
		hvec = np.logspace(-14,-1,100)

	max_err = 0	
	J = jacobian(x)
	print("J", J)
	#print "residual norm", norm(residual(x))

	for i in range(n):
		ei = np.zeros(x.shape, dtype = np.float)
		ei[i] = 1.
		err = np.zeros(len(hvec))
		Ji_est = np.zeros((len(hvec), len(J[:,i])))
		for k,h in enumerate(hvec):
			x1 = x + ei * h
			x2 = x - ei * h
			f1 = residual(x1) 
			f2 = residual(x2)
			print('x', x1,', f(x)',  f1)
			print('x', x2,', f(x)',  f2)
			Ji_est[k,:] = (f1 - f2)/(2*h)
			err[k] = np.linalg.norm(Ji_est[k,:] - J[:,i], 2)
			
		#err = [ np.linalg.norm( (residual(x+ei*h) - residual(x-ei*h))/(2*h) - J[:,i], 2) for h in hvec]
		k = np.argmin(err)
		print("%3d: nominal norm:%5.5e, err:%5.5e, h: %3.3e" % (i, np.linalg.norm(J[:,i]), err[k], hvec[k]))
		print("J    [:,i]", J[:,i])
		print("J_est[:,i]", Ji_est[k,:])
		max_err = max(min(err)/norm(J[:,i]), max_err)

	return max_err


def check_gradient(x, residual, jacobian, hvec = None):
	n = x.shape[0]
	if hvec is None:
		hvec = np.logspace(-14,-1,100)
	

	max_err = 0	
	J = jacobian(x)
	r = residual(x)
	g = np.dot(J.T, r) 

	for i in range(n):
		ei = np.zeros(x.shape, dtype = np.float)
		ei[i] = 1.
		err = [ np.abs( 0.5*( norm(residual(x+ei*h))**2 - norm(residual(x-ei*h))**2)/(2*h) - g[i]) for h in hvec]
		max_err = max(min(err), max_err)
		print("%3d: err:%5.5e" % (i, min(err)))

	return max_err
		
def check_derivative(x, obj, grad, hvec = None):
	n = x.shape[0]
	if hvec is None:
		hvec = np.logspace(-14,-1,100)

	max_err = 0	
	g = grad(x)
	for i in range(n):
		ei = np.zeros(x.shape, dtype = np.float)
		ei[i] = 1.
		err = [ np.linalg.norm( ( obj(x+ei*h) - obj(x-ei*h) )/(2*h) - g[i], np.inf) for h in hvec]
		max_err = max(np.min(err), max_err)
		print("%3d: err:%5.5e" % (i, np.min(err)))

	return max_err


def check_hessian(x, obj, hess):
	n = x.shape[0]
	H = hess(x)

	hvec = np.logspace(-7,-1,10)
	max_err = 0.

	for i in range(n):	
		ei = np.zeros(n)
		ei[i] = 1.
		for j in range(n):
			ej = np.zeros(n)
			ej[j] = 1.
			min_err = np.inf
			for h in hvec:
				Hij_est = ( (obj(x + h*ei + h*ej) - obj(x - h*ei + h*ej)) - (obj(x + h*ei - h*ej) - obj(x - h*ei - h*ej)) )/(4*h**2)
				err = np.max(np.abs(Hij_est - H[i,j]))
				min_err = np.min([min_err, err])
			print("%d %d %5.2e : %5.2e %5.2e" % (i,j,min_err, H[i,j], Hij_est))
			max_err = max(min_err, max_err)
	return max_err

#if __name__ == '__main__':
#	z = np.exp(2j*np.pi*np.linspace(0,1, 1000, endpoint = False))
#	h = np.tan(64*z)
#
#	n = 4
#	pb = PolynomialBasisRationalFit(n,n)
#	#pb = PoleResidueRationalFit(n,n, real = False)
#	pb.fit(z, h)
#	#b = pb._init_aaa()
#	b = np.random.randn(n+1) + 1j*np.random.randn(n+1)
#	print b
#
#	residual = lambda x: pb.residual(x.view(complex), return_real = True)
#	jacobian = lambda x: pb.jacobian(x.view(complex))
#	check_jacobian(b.view(float), residual, jacobian)
