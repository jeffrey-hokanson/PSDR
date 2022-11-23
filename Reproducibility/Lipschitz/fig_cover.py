import numpy as np
import psdr
from psdr.pgf import PGF

np.random.seed(0)

m = 2
domain = psdr.BoxDomain(-np.ones(m), np.ones(m))
L = np.array([[1, 0],[-1,4]])/4

print("without Lipschitz")
X = psdr.minimax_lloyd(domain, 9, maxiter = 500)
print(X)
d = psdr.geometry.fill_distance(domain, X)
print(d) 

pgf = PGF()
pgf.add('x', X[:,0])
pgf.add('y', X[:,1])
pgf.write('data/fig_cover_scalar.dat')


print("with Lipschitz")
X = psdr.minimax_lloyd(domain, 3, L = L)
print(X)
d = psdr.geometry.fill_distance(domain, X, L = L)
print(d) 

pgf = PGF()
pgf.add('x', X[:,0])
pgf.add('y', X[:,1])
pgf.write('data/fig_cover_matrix.dat')
