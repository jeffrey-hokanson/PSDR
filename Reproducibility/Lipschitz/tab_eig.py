import numpy as np
import scipy.linalg
import psdr
import psdr.demos
from psdr.pgf import PGF

np.random.seed(0)

funs = []
names = []

# OTL circuit function
funs.append(psdr.demos.OTLCircuit())
names.append('otl')

# Borehole
funs.append(psdr.demos.Borehole())
names.append('borehole')



act = psdr.ActiveSubspace()
lip = psdr.LipschitzMatrix()


m = max([len(fun.domain) for fun in funs])
pgf = PGF()
pgf.add('i', np.arange(1,m+1))
for fun, name in zip(funs, names):
	X = fun.domain.sample(2e2)
	grads = fun.grad(X)
	
	act.fit(grads)
	lip.fit(grads = grads)

	ew = np.nan*np.zeros(m)
	ew[0:len(fun.domain)] = scipy.linalg.eigvalsh(act.C)[::-1]
	pgf.add('%s_C' % name, ew)
	ew[0:len(fun.domain)] = scipy.linalg.eigvalsh(lip.H)[::-1]
	pgf.add('%s_H' % name, ew)

pgf.write('data/tab_eigs.dat')
