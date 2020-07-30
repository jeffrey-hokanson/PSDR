import numpy as np
import psdr, psdr.demos

np.random.seed(0)

#fun = psdr.demos.OTLCircuit()
fun = psdr.demos.Borehole()

X = fun.domain.sample(1e2)
grads = fun.grad(X)

lip = psdr.LipschitzMatrix(verbose = True)
lip.fit(grads = grads)


U = lip.U[:,0:1]
print(U)

# setup an experimental design
X = psdr.minimax_design_1d(fun.domain, 20, L = U.T)
X2 = []
for x in X:
	dom = fun.domain.add_constraints(A_eq = U.T, b_eq = U.T @ x)
	X2.append(dom.sample_boundary(50))

X2 = np.vstack(X2)

fX2 = fun(X2)

pra = psdr.PolynomialRidgeApproximation(9, 1, norm = np.inf)

pra.fit_fixed_subspace(X2, fX2, U)


if True:
	X3 = psdr.minimax_design_1d(fun.domain, 500, L = U.T)
	yapprox = pra(X3)
	y = fun(X3)
	
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	ax.plot((U.T @ X2.T).flatten(), fX2, 'k.')
	ax.plot((U.T @ X3.T).flatten(), yapprox, 'r-')
	
	plt.show()

