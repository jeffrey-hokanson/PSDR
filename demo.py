import numpy as np
import psdr
import psdr.demos
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orth
if True:
	np.random.seed(0)
	fun = psdr.demos.OTLCircuit()
	X = fun.domain.sample(5)
	fX = fun(X)
	Xg = fun.domain.sample(20)
	grads = fun.grad(Xg)

	lip = psdr.LowRankLipschitzMatrix(3, verbose = True)
	lip.fit(X, fX, grads)
	H = lip.H
	print(H)
	print(np.linalg.eigvalsh(H))
#	U = lip._init_U(X, fX, grads)
#	print(U)
#	act = psdr.ActiveSubspace()
#	act.fit(grads = grads)

#	U = act.U[:,0:2]
#	#U = np.random.randn(*U.shape)
	#U = orth(U)
	# U = np.random.randn(len(fun.domain),6)
#	grads = grads[0:50]
	#J, alpha = lip._fixed_U(U, X, fX, grads, 0)
	#print(J, np.linalg.eigvals(J))
	#print(alpha)

	#lip._U_descent(U, J, alpha, X, fX, grads, 0)
	#lip._optimize(U, X, fX, grads, 0)
