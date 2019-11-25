import numpy as np
import psdr
from psdr.vandermonde import *
from psdr.basis import index_set

# NB: This code is too slow to use in testing;
# I also don't like adding multiprecision to testing
def test_vandermonde_arnoldi_mp(p=5):
	from mpmath import mp
	X = np.random.rand(100,3)
	Q, H = vandermonde_arnoldi(X, p)

	# Compute Q matrix using multiprecision
	idx = index_set(p, X.shape[1])
	with mp.workdps(100):
		Xm = mp.matrix(X)
		V = mp.ones(X.shape[0], len(idx))
		for k, ids in enumerate(idx):
			for j, d in enumerate(ids):
				for i in range(X.shape[0]):
					V[i,k] *= Xm[i,j]**int(d)
		Qe, R = mp.qr(V, mode = 'skinny')

		Qm = mp.matrix(Q/np.sqrt(X.shape[0]))
		
		A = Qe.T * Qm
		
		s = mp.svd_r(A, compute_uv = False)
		print(s[-1])

def test_polyfitA(m = 2):
	
	# Multi-dimensional Runge function
	f = lambda X: 1./(1 + 25*np.sum(X**2, axis = 1))
#	f = lambda X: np.sum(X**5, axis = 1) + X[:,0]**2*X[:,1]**3
#	f = lambda X: np.sum(np.cos(X), axis = 1)

	# Testing data for sup-norm
	n = int(np.ceil( 1e4**(1/m)))
	x = np.cos(np.arange(n+1)*np.pi/n)
	Xs = np.meshgrid(*[x for i in range(m)])
	Xt = np.vstack([X.flatten() for X in Xs]).T	
	fXt = f(Xt)	

	
	# Instead we simply do random
	for p in range(40, 200):
		if m  == 1:
			gs = p + 1
		else:	
			gs = np.ceil(len(index_set(p, m))**(1./m))*2
		x = np.cos(np.arange(gs+1)*np.pi/gs)
		Xs = np.meshgrid(*[x for i in range(m)], indexing = 'ij')
		X = np.vstack([X.flatten() for X in Xs]).T

		# Evaluate the function
		fX = f(X)

		d, H = polyfitA(X, fX, p)
		y = polyvalA(d, H, Xt)

		# Compare against standard
		pa = psdr.PolynomialApproximation(degree = p, basis = 'legendre')
		pa.fit(X, fX)	
		y_leg = pa(Xt)
		print(len(fX), len(index_set(p, m)))	
		print(p, '%10.6e' % np.linalg.norm(fXt - y, np.inf), '%10.6e' % np.linalg.norm(fXt - y_leg, np.inf))
		


if __name__ == "__main__":
	test_polyfitA()
