import numpy as np
from numpy.linalg import norm

def check_jacobian(x, residual, jacobian):
	n = x.shape[0]
	hvec = np.logspace(-14,-1,100)

	max_err = 0	
	J = jacobian(x)

	#print "residual norm", norm(residual(x))

	for i in range(n):
		ei = np.zeros(x.shape, dtype = np.float)
		ei[i] = 1.
		err = [ np.linalg.norm( (residual(x+ei*h) - residual(x-ei*h))/(2*h) - J[:,i], 2) for h in hvec]
		j = np.argmin(err)
		print "%3d: nominal norm:%5.5e, err:%5.5e, h: %3.3e" % (i, np.linalg.norm(J[:,i]), min(err), hvec[j])
		Jest = (residual(x + ei*hvec[j]) - residual(x - ei*hvec[j]))/(2*hvec[j])
		#print "Jest", Jest[0:4]
		#print "J   ", J[0:4,i] 
		max_err = max(min(err)/norm(J[:,i]), max_err)

	return max_err

def check_gradient(x, residual, jacobian):
	n = x.shape[0]
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
		print "%3d: err:%5.5e" % (i, min(err))

	return max_err
		
def check_derivative(x, obj, grad):
	n = x.shape[0]
	hvec = np.logspace(-14,-1,100)

	max_err = 0	
	g = grad(x)
	for i in range(n):
		ei = np.zeros(x.shape, dtype = np.float)
		ei[i] = 1.
		err = [ np.abs( ( obj(x+ei*h) - obj(x-ei*h) )/(2*h) - g[i]) for h in hvec]
		max_err = max(min(err), max_err)
		print "%3d: err:%5.5e" % (i, min(err))

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
