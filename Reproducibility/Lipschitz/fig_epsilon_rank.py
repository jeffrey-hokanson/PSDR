import numpy as np
import psdr, psdr.demos
from psdr.pgf import PGF

from joblib import Memory
memory = Memory('.cache', verbose = False)
fun = psdr.demos.Borehole()




if False:
	fun = psdr.demos.HartmannMHD()
	X = fun.domain.sample_grid(4)

	fX = fun(X)[:,0]

fun = psdr.demos.OTLCircuit()

np.random.seed(0)

@memory.cache
def generate_X(fun, M):
	np.random.seed(0)
	X = fun.domain.sample(1000)
	grads = fun.grad(X)
	lip = psdr.LipschitzMatrix()
	lip.fit(grads = grads)
	L = lip.L
	
	X = psdr.minimax_lloyd(fun.domain, M, L = L, verbose = True)
	return X

@memory.cache
def generate_X_maximin(fun, M):
	np.random.seed(0)
	X = fun.domain.sample(1000)
	grads = fun.grad(X)
	lip = psdr.LipschitzMatrix(verbose = True, abstol = 1e-7, reltol = 1e-7, feastol = 1e-7)
	lip.fit(grads = grads)
	L = lip.L

	X0 = psdr.maximin_coffeehouse(fun.domain, M, L = L, verbose =True)		
	X = psdr.maximin_block(fun.domain, M, L = L, verbose = True, Xhat = X0)
	return X

X = generate_X_maximin(fun, 200)
fX = fun(X)

print(len(fX))
	

fX_range = np.max(fX) - np.min(fX)

@memory.cache
def lipschitz(X, fX, epsilon):
	lip = psdr.LipschitzMatrix(epsilon = epsilon, verbose = True, abstol = 1e-7, reltol = 1e-7, feastol = 1e-7)
	lip.fit(X, fX)
	return lip.L


epsilon = np.linspace(0, fX_range, 81)
rank = np.zeros(epsilon.shape, dtype = np.int)
obj = np.zeros(epsilon.shape)
lam1 = np.zeros(epsilon.shape)
lam2 = np.zeros(epsilon.shape)
lam3 = np.zeros(epsilon.shape)
lam4 = np.zeros(epsilon.shape)
lam5 = np.zeros(epsilon.shape)
lam6 = np.zeros(epsilon.shape)


for k, eps in enumerate(epsilon):
	L = lipschitz(X, fX, eps)
	rank[k] = np.linalg.matrix_rank(L)
	obj[k] = np.linalg.norm(L, 'fro')
	ew, ev = np.linalg.eigh(L)
	ew = ew[::-1]
	lam1[k] = ew[0]
	lam2[k] = ew[1]
	lam3[k] = ew[2]
	lam4[k] = ew[3]
	lam5[k] = ew[4]
	lam6[k] = ew[5]
	print(f"=====> epsilon {eps:8.2e}, rank {rank[k]:2d}, obj {obj[k]:10.5e}") 

pgf = PGF()
pgf.add('epsilon', epsilon)
pgf.add('rank', rank)
pgf.add('obj', obj)
pgf.add('lam1', lam1)
pgf.add('lam2', lam2)
pgf.add('lam3', lam3)
pgf.add('lam4', lam4)
pgf.add('lam5', lam5)
pgf.add('lam6', lam6)
pgf.write('data/fig_epsilon_rank.dat')
