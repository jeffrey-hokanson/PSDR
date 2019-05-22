from __future__ import print_function
import filecmp
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import psdr, psdr.demos

def test_shadow():
	fun = psdr.demos.OTLCircuit()
	X = fun.domain.sample_grid(4)
	fX = fun(X)
	
	Xg = fun.domain.sample_grid(6)
	fXg = fun(Xg)

	pra = psdr.PolynomialRidgeApproximation(degree = 5, subspace_dimension = 1)
	pra.fit(X, fX)


	# Generate shadow plot with response surface
	ax = pra.shadow_plot(X, fX, pgfname = 'test_shadow.dat')

	assert filecmp.cmp('data/test_shadow.dat', 'test_shadow.dat') 
	assert filecmp.cmp('data/test_shadow_response.dat', 'test_shadow_response.dat') 


	# Generate shadow envelope
	pra.shadow_envelope(Xg, fXg, ax = ax, pgfname = 'test_shadow_envelope.dat')
	assert filecmp.cmp('data/test_shadow_envelope.dat', 'test_shadow_envelope.dat') 

	fig, ax2 = plt.subplots()

	pra2 = psdr.PolynomialRidgeApproximation(degree = 5, subspace_dimension = 2)
	pra2.fit(X, fX)
	pra2.shadow_plot(X, fX, ax = ax2)

def test_shadow_lipschitz():
	fun = psdr.demos.OTLCircuit()
	X = fun.domain.sample_grid(2)
	fX = fun(X)
	grads = fun.grad(X)

	lip = psdr.LipschitzMatrix()
	lip.fit(grads = grads)	
	
	ax = lip.shadow_plot(X, fX)
	lip.shadow_uncertainty(fun.domain, X, fX, ax = ax, ngrid = 4, pgfname = 'test_shadow_uncertainty.dat')
	assert filecmp.cmp('data/test_shadow_uncertainty.dat', 'test_shadow_uncertainty.dat') 


if __name__ == '__main__':
	#test_shadow_lipschitz()
	test_shadow()
	plt.show()
	

