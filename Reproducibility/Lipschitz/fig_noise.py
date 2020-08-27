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
np.random.seed(0)
v0 = np.random.randn(A.shape[0])
B = np.eye(A.shape[0])[:,0:2]
#B = 10*np.random.randn(A.shape[0], 2)


def partial_trace_fun(x, tol = 0.1):
	ew = eigsh(A + diags(B @ x), which = 'SM', v0 = v0, tol = tol, return_eigenvectors = False)
	return float(np.sum(ew))

domain = psdr.BoxDomain([0,0], [1,1])
fun = psdr.Function(partial_trace_fun, domain)

fun_true = psdr.Function(lambda x: partial_trace_fun(x, tol = 1e-4), domain)

X = fun.domain.sample_grid(10)
fX = fun(X)
fX_true = fun_true(X)


epsilon = float(max(np.abs(fX - fX_true)))


lip = psdr.LipschitzMatrix(verbose = True)
lip.fit(X, fX)
print("noisy", lip.L)
print(np.linalg.norm(lip.L, 'fro'))

lip.fit(X, fX_true)
print("true", lip.L)
print(np.linalg.norm(lip.L, 'fro'))

print("epsilon", epsilon)
print("max fX true", np.max(fX_true))

lip = psdr.LipschitzMatrix(verbose = True, epsilon = 2*epsilon)
lip.fit(X, fX)
print("epsilon-Lipschitz", lip.L)
print(np.linalg.norm(lip.L, 'fro'))

if True:
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2)
	tc0 = ax[0].tricontourf(X[:,0], X[:,1], fX[:,0], levels = 20)
	tc1 = ax[1].tricontourf(X[:,0], X[:,1], fX_true[:,0], levels = 20)
	fig.colorbar(tc0, ax=ax[0])
	fig.colorbar(tc1, ax=ax[1])
	plt.show()
