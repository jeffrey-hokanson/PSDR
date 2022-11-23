from __future__ import print_function

import numpy as np

import psdr
import psdr.demos
from psdr.pgf import PGF

np.random.seed(0)

# Approximate number of grid points to use for verification
Ngrid = int(1e6)

# Number of points to sample
M = 100

# Volume of Golinski Gearbox
gg = psdr.demos.GolinskiGearbox()
gg._funs = [gg._funs[0]]
gg._grads = [gg._grads[0]]

# other functions
otl = psdr.demos.OTLCircuit()
piston = psdr.demos.Piston()
borehole = psdr.demos.Borehole()
wing = psdr.demos.WingWeight()

for fun, name in zip([gg, otl, piston, borehole, wing],['golinski', 'otl', 'piston', 'borehole', 'wing']):
#for fun, name in zip([otl],['otl']):
	# Construct a grid for testing purposes
	ngrid = int(np.ceil(Ngrid**(1./len(fun.domain))))
	Xg = fun.domain.sample_grid(ngrid)

	# Compute the Lipschitz matrix from the corners of the domain
	X = fun.domain.sample_grid(2)
	gradX = fun.grad(X)

	lip_mat = psdr.LipschitzMatrix()
	lip_mat.fit(grads = gradX)
	lip_con = psdr.LipschitzConstant()
	lip_con.fit(grads = gradX)
		
	# scale by function variation
	fXg = fun(Xg)
	#uncertain = (ub - lb)/(np.max(fXg) - np.min(fXg))
	rnge = np.max(fXg) - np.min(fXg) 

	X_iso = psdr.minimax_lloyd(fun.domain, M)
	X_lip = psdr.minimax_lloyd(fun.domain, M, L = lip_mat.L)
	

	# Isotropic sampling/ scalar Lipschitz	
	lb, ub = lip_con.uncertainty(X_iso, fun(X_iso), Xg)
	p = np.percentile(ub - lb, [0,25,50, 75,100])
	print(p)
	pgf = PGF()
	for i, t in enumerate([0,25, 50, 75, 100]):
		pgf.add('p%d' % t, [p[i]])
	pgf.add('range', [rnge])
	pgf.write('data/tab_sample_%s_scalar_isotropic_uncertainty.dat' % (name) ) 	
	

	# Isotropic sampling/ matrix Lipschitz	
	lb, ub = lip_mat.uncertainty(X_iso, fun(X_iso), Xg)
	p = np.percentile(ub - lb, [0,25,50, 75,100])
	print(p)
	pgf = PGF()
	for i, t in enumerate([0,25, 50, 75, 100]):
		pgf.add('p%d' % t, [p[i]])
	pgf.add('range', [rnge])
	pgf.write('data/tab_sample_%s_matrix_isotropic_uncertainty.dat' % (name) ) 	
	
	# Isotropic sampling/ matrix Lipschitz	
	lb, ub = lip_mat.uncertainty(X_lip, fun(X_lip), Xg)
	p = np.percentile(ub - lb, [0,25,50, 75,100])
	print(p)
	pgf = PGF()
	for i, t in enumerate([0,25, 50, 75, 100]):
		pgf.add('p%d' % t, [p[i]])
	pgf.add('range', [rnge])
	pgf.write('data/tab_sample_%s_matrix_lipschitz_uncertainty.dat' % (name) ) 	
