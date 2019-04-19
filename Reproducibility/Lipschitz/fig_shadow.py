import numpy as np
import psdr
import psdr.demos

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
	print("computing envelope")
	lip.shadow_envelope(Xg, fXg, ax = None, pgfname = 'data/fig_shadow_%s_envelope.dat' % (name, ), U = U)
	
	print("computing envelope estimate")
	lip.shadow_envelope_estimate(fun.domain, X, fX, U = U,
		pgfname = 'data/fig_shadow_%s_envelope_estimate.dat' % (name,), progress = 2, ngrid = 100)
