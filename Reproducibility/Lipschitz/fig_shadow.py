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

lip_mat.fit(grads = gradX)
lip_con.fit(grads = gradX)

# construct designs
M = 100
ngrid = 100
X_iso = psdr.minimax_lloyd(fun.domain, M)
X_lip = psdr.minimax_lloyd(fun.domain, M, L = lip_mat.L)

# Fix ridge
U = lip_mat.U[:,0].copy()

# Generate envelope of data
lip_con.shadow_envelope(Xg, fXg, ax = None, pgfname = 'data/fig_shadow_envelope.dat',  U = U)

# isotropic / scalar
lip_con.shadow_plot(X_iso, fun(X_iso), ax = None, U = U, dim = 1, pgfname = 'data/fig_shadow_iso_scalar.dat') 
lip_con.shadow_uncertainty(fun.domain, X_iso, fun(X_iso), U = U,
	pgfname = 'data/fig_shadow_iso_scalar_shadow_uncertainty.dat',  progress = 2, ngrid = ngrid)

# isotropic / matrix
lip_mat.shadow_plot(X_iso, fun(X_iso), ax = None, U = U, dim = 1, pgfname = 'data/fig_shadow_iso_matrix.dat') 
lip_mat.shadow_uncertainty(fun.domain, X_iso, fun(X_iso), U = U,
	pgfname = 'data/fig_shadow_iso_matrix_shadow_uncertainty.dat',  progress = 2, ngrid = ngrid)

# lipschitz / matrix
lip_mat.shadow_plot(X_lip, fun(X_lip), ax = None, U = U, dim = 1, pgfname = 'data/fig_shadow_lip_matrix.dat') 
lip_mat.shadow_uncertainty(fun.domain, X_lip, fun(X_lip), U = U,
	pgfname = 'data/fig_shadow_lip_matrix_shadow_uncertainty.dat',  progress = 2, ngrid = ngrid)

#
#for lip, name in zip([lip_mat, lip_con], ['mat', 'con']):
#	lip.fit(grads = gradX)
##	if name == 'mat':
##		U = lip.U[:,0].copy()
##		pgf = PGF()
##		for i, u in enumerate(U):
##			pgf.add('x%d' % (i+1), [u])
##		pgf.write('data/fig_shadow_U.dat')
#
#	lip.shadow_plot(X, fX, ax = None, U = U, dim = 1, pgfname = 'data/fig_shadow_%s.dat' % (name,) ) 
#	
#	if True:
#		print("computing envelope")
#		lip.shadow_envelope(Xg, fXg, ax = None, pgfname = 'data/fig_shadow_%s_envelope.dat' % (name, ), U = U)
#
#	if True:	
#		print("computing shadow uncertainty")
#		lip.shadow_uncertainty(fun.domain, X, fX, U = U,
#			pgfname = 'data/fig_shadow_%s_shadow_uncertainty.dat' % (name,), progress = 2, ngrid = 100)
