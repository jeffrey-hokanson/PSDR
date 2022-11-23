import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from scipy.io import loadmat

import psdr

mat = loadmat('Trefethen_700.mat')
# Extract the matrix
A = mat['Problem'][0][0][2]

# Initial random vector
v0 = np.ones(A.shape[0])
B = np.eye(A.shape[0])[:,0:2]
#B = 10*np.random.randn(A.shape[0], 2)


def partial_trace_fun(x, tol = 0.10):
	Ax = A + 2.5 * diags(B @ (x + 1) )
	ew = eigsh(Ax, k = 5, which = 'SM', v0 = v0, tol = tol, return_eigenvectors = False)
	return float(np.sum(ew))

domain = psdr.BoxDomain([-1,-1], [1,1])
fun = psdr.Function(partial_trace_fun, domain)

fun_true = psdr.Function(lambda x: partial_trace_fun(x, tol = 1e-4), domain)

X = fun.domain.sample_grid(10)
#np.random.seed(0)
#X = fun.domain.sample(200)
fX = fun(X)
fX_true = fun_true(X)


epsilon = float(max(np.abs(fX - fX_true)))
#epsilon = 0.2*np.max(np.abs(fX_true))
#print(np.max(fX_true))
#print(epsilon)

lip = psdr.LipschitzMatrix(verbose = True)
lip.fit(X, fX)
print("========noisy=========")
print("L", lip.L)
print(np.linalg.norm(lip.L, 'fro'))
ew, ev = np.linalg.eigh(lip.L)
print("ew", ew)
print("ev", ev)

lip.fit(X, fX_true)
print("========true=========")
print("L", lip.L)
print(np.linalg.norm(lip.L, 'fro'))
ew, ev = np.linalg.eigh(lip.L)
print("ew", ew)
print("ev", ev)


lip = psdr.LipschitzMatrix(verbose = True, epsilon = epsilon)
lip.fit(X, fX)
print("========epsilon=========")
print("L", lip.L)
print("epsilon-Lipschitz", lip.L)
print(np.linalg.norm(lip.L, 'fro'))
ew, ev = np.linalg.eigh(lip.L)
print("ew", ew)
print("ev", ev)

print("epsilon", epsilon)
print("max fX true", np.max(fX_true))

if False:
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2)
	tc0 = ax[0].tricontourf(X[:,0], X[:,1], fX[:,0], levels = 20)
	tc1 = ax[1].tricontourf(X[:,0], X[:,1], fX_true[:,0], levels = 20)
	fig.colorbar(tc0, ax=ax[0])
	fig.colorbar(tc1, ax=ax[1])
	plt.show()
