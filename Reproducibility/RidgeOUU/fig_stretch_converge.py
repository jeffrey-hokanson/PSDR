import numpy as np
import psdr, psdr.demos

fun = psdr.demos.GolinskiGearbox()

domain = fun.domain

X = domain.sample(10)
fX = fun(X)


pras = [psdr.PolynomialRidgeApproximation(degree = 2, subspace_dimension = 1)]*fX.shape[1]

for it in range(500):
	Ls = []
	for k, pra in enumerate(pras):
		pra.fit(X, fX[:,k])
		Ls.append(pra.U.T)
		
	
	x = psdr.stretch_sample(domain, X, Ls, verbose = False)
	print(f'it {it:4d}|  '+ '   '.join([f'{xi:8.3f}' for xi in x]))
