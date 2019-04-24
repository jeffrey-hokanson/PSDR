import numpy as np
import psdr
import psdr.demos
from psdr.pgf import PGF

np.random.seed(0)

fun = psdr.demos.OTLCircuit()

X = fun.domain.sample_grid(2)
fX = fun(X)
gradX = fun.grad(X)

# Grid-based sampling
Xg = fun.domain.sample_grid(8)
fXg = fun(Xg)

lip_mat = psdr.LipschitzMatrix()
lip_con = psdr.LipschitzConstant()


for lip, name in zip([lip_mat, lip_con], ['mat', 'con']):
	lip.fit(grads = gradX)
	if name == 'mat':
		U = lip.U[:,0].copy()
		pgf = PGF()
		for i, u in enumerate(U):
			pgf.add('x%d' % (i+1), [u])
		pgf.write('data/fig_shadow_U.dat')

	lip.shadow_plot(X, fX, ax = None, U = U, dim = 1, pgfname = 'data/fig_shadow_%s.dat' % (name,) ) 
	
	if True:
		print("computing envelope")
		lip.shadow_envelope(Xg, fXg, ax = None, pgfname = 'data/fig_shadow_%s_envelope.dat' % (name, ), U = U)

	if True:	
		print("computing envelope estimate")
		lip.shadow_envelope_estimate(fun.domain, X, fX, U = U,
			pgfname = 'data/fig_shadow_%s_envelope_estimate.dat' % (name,), progress = 2, ngrid = 100)
