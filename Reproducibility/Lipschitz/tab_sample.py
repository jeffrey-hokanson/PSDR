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

#for fun, name in zip([gg, otl, piston, borehole, wing],['golinski', 'otl', 'piston', 'borehole', 'wing']):
for fun, name in zip([otl],['otl']):
	# Construct a grid for testing purposes
	ngrid = int(np.ceil(Ngrid**(1./len(fun.domain))))
	Xg = fun.domain.sample_grid(ngrid)

	# Compute the Lipschitz matrix from the corners of the domain
	X = fun.domain.sample_grid(2)
	gradX = fun.grad(X)

	lip_mat = psdr.LipschitzMatrix()
	lip_con = psdr.LipschitzConstant()
	
	for lip, lip_name in zip([lip_mat, lip_con], ['mat', 'con']):
		lip.fit(grads = gradX)

		if lip_name == 'mat':
			U = lip.U.copy()
	
		# Now perform sequential sampling 
		samp = psdr.SequentialMaximinSampler(fun, lip.L)
		samp.sample(M, verbose = True)
		
		# Compute error bounds
		lb, ub = lip.bounds(samp.X, samp.fX, Xg)

		# scale by function variation
		fXg = fun(Xg)
		uncertain = (ub - lb)/(np.max(fXg) - np.min(fXg))
	
		p = np.percentile(uncertain, [0,25,50, 75,100])
		print(p)
		pgf = PGF()
		for i, t in enumerate([0,25, 50, 75, 100]):
			pgf.add('p%d' % t, [p[i]])
		pgf.write('data/tab_sample_%s_%s_uncertainty.dat' % (name, lip_name) ) 	

		if name == 'otl':	
			Nsamp = 64	
			lip.shadow_plot(samp.X[:Nsamp], samp.fX[:Nsamp], ax = None, 
				U = U, dim = 1, pgfname = 'data/tab_sample_%s_%s_shadow.dat' % (name, lip_name) ) 

			if False:	
				print("computing envelope estimate")
				lip.shadow_envelope_estimate(fun.domain, samp.X[:Nsamp], samp.fX[:Nsamp], 
					pgfname = 'data/tab_sample_%s_%s_envelope_estimate.dat' % (name, lip_name), 
					progress = 2, ngrid = 100, U = U[:,0])

			if lip_name == 'con':
				# The actual envelope is indepent of the Lipschitz matrix/constant 
				Xg2 = fun.domain.sample_grid(8)
				fXg2 = fun(Xg2)
				print("computing envelope")
				lip.shadow_envelope(Xg2, fXg2, ax = None, 
					pgfname = 'data/tab_sample_%s_envelope.dat' % (name, ), U = U[:,0])
